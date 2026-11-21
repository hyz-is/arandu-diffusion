package diffusion

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg" // registers the JPEG decoder with image.Decode
	"image/png"
)

// pixels is an image as the library takes it: interleaved, row-major, one byte
// per channel. RGB has three channels and a mask has one.
type pixels struct {
	w, h, channels int
	data           []byte
}

// decodeConfig reads the size of an encoded image without decoding it, so a
// decompression bomb is refused for its header and not after it has been
// allocated.
func decodeConfig(b []byte) (image.Config, error) {
	cfg, _, err := image.DecodeConfig(bytes.NewReader(b))
	if err != nil {
		return cfg, fmt.Errorf("%w: image: %v", ErrInvalidRequest, err)
	}
	if cfg.Width <= 0 || cfg.Height <= 0 || cfg.Width*cfg.Height > MaxImagePixels {
		return cfg, fmt.Errorf("%w: image of %dx%d is outside the limit of %d pixels", ErrInvalidRequest, cfg.Width, cfg.Height, MaxImagePixels)
	}
	return cfg, nil
}

// decodeRGB decodes a PNG or JPEG into RGB. Transparency is flattened onto
// white, which is what a photograph of the object on a page would show.
func decodeRGB(b []byte) (pixels, error) {
	if _, err := decodeConfig(b); err != nil {
		return pixels{}, err
	}
	img, _, err := image.Decode(bytes.NewReader(b))
	if err != nil {
		return pixels{}, fmt.Errorf("%w: image: %v", ErrInvalidRequest, err)
	}
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	flat := image.NewNRGBA(image.Rect(0, 0, w, h))
	draw.Draw(flat, flat.Bounds(), image.NewUniform(color.White), image.Point{}, draw.Src)
	draw.Draw(flat, flat.Bounds(), img, bounds.Min, draw.Over)
	out := pixels{w: w, h: h, channels: 3, data: make([]byte, w*h*3)}
	for y := range h {
		row := flat.Pix[y*flat.Stride : y*flat.Stride+w*4]
		dst := out.data[y*w*3 : (y+1)*w*3]
		for x := range w {
			dst[x*3], dst[x*3+1], dst[x*3+2] = row[x*4], row[x*4+1], row[x*4+2]
		}
	}
	return out, nil
}

// decodeMask decodes a PNG or JPEG into one luminance channel. Transparent
// pixels count as black, that is, as kept.
func decodeMask(b []byte) (pixels, error) {
	if _, err := decodeConfig(b); err != nil {
		return pixels{}, err
	}
	img, _, err := image.Decode(bytes.NewReader(b))
	if err != nil {
		return pixels{}, fmt.Errorf("%w: mask: %v", ErrInvalidRequest, err)
	}
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	out := pixels{w: w, h: h, channels: 1, data: make([]byte, w*h)}
	for y := range h {
		for x := range w {
			// GrayModel premultiplies by alpha, so a transparent pixel is 0.
			out.data[y*w+x] = color.GrayModel.Convert(img.At(bounds.Min.X+x, bounds.Min.Y+y)).(color.Gray).Y
		}
	}
	return out, nil
}

// resolveSize is the pixel size a request will run at: the explicit size, else
// the size of the image it starts from, else 512.
func resolveSize(req Request, init *pixels) (w, h int, err error) {
	w, h = req.Width, req.Height
	if init != nil {
		if w == 0 {
			w = init.w
		}
		if h == 0 {
			h = init.h
		}
		if w != init.w || h != init.h {
			return 0, 0, fmt.Errorf("%w: the size %dx%d differs from the init image, %dx%d; resize the image first", ErrInvalidRequest, w, h, init.w, init.h)
		}
		if w%8 != 0 || h%8 != 0 || w < MinSide || h < MinSide || w > MaxSide || h > MaxSide {
			return 0, 0, fmt.Errorf("%w: the init image is %dx%d; its sides must be multiples of 8 between %d and %d", ErrInvalidRequest, w, h, MinSide, MaxSide)
		}
		return w, h, nil
	}
	if w == 0 {
		w = 512
	}
	if h == 0 {
		h = 512
	}
	return w, h, nil
}

// prepared is a request with its images decoded and its size settled, so the
// expensive step starts from inputs already known to be good.
type prepared struct {
	req   Request
	mode  Mode
	w, h  int
	seed  int64
	steps int
	init  *pixels
	mask  *pixels
	refs  []pixels
}

// prepare validates the request, decodes every image and checks them against
// each other. It does all the work that can fail for the caller's reasons, so
// whatever fails after it fails for the machine's.
func prepare(req Request, seed func() (int64, error)) (*prepared, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	p := &prepared{req: req, mode: req.Mode(), steps: req.Steps}
	if p.steps == 0 {
		p.steps = 20
	}
	if len(req.Init) > 0 {
		img, err := decodeRGB(req.Init)
		if err != nil {
			return nil, err
		}
		p.init = &img
	}
	if len(req.Mask) > 0 {
		m, err := decodeMask(req.Mask)
		if err != nil {
			return nil, err
		}
		if m.w != p.init.w || m.h != p.init.h {
			return nil, fmt.Errorf("%w: the mask is %dx%d and the image is %dx%d", ErrInvalidRequest, m.w, m.h, p.init.w, p.init.h)
		}
		p.mask = &m
	}
	for _, ref := range req.References {
		img, err := decodeRGB(ref)
		if err != nil {
			return nil, err
		}
		p.refs = append(p.refs, img)
	}
	var err error
	if p.w, p.h, err = resolveSize(req, p.init); err != nil {
		return nil, err
	}
	p.seed = req.Seed
	if p.seed < 0 {
		if p.seed, err = seed(); err != nil {
			return nil, err
		}
	}
	return p, nil
}

// composite puts the original back wherever the mask keeps it: a pixel whose
// mask is 0 is copied from the original byte for byte, a pixel whose mask is 255
// is the generated one, and anything between is a blend.
//
// The library keeps the masked area close to the original but not equal to it:
// the whole image goes through the autoencoder, and the autoencoder is lossy.
// Equality is a promise this package makes, so it is made here, after the
// library, where it can be checked.
func composite(original, generated pixels, mask pixels) pixels {
	out := pixels{w: original.w, h: original.h, channels: 3, data: make([]byte, len(original.data))}
	for i := range mask.data {
		switch m := int(mask.data[i]); m {
		case 0:
			copy(out.data[i*3:i*3+3], original.data[i*3:i*3+3])
		case 255:
			copy(out.data[i*3:i*3+3], generated.data[i*3:i*3+3])
		default:
			for c := range 3 {
				o, g := int(original.data[i*3+c]), int(generated.data[i*3+c])
				out.data[i*3+c] = byte((o*(255-m) + g*m + 127) / 255)
			}
		}
	}
	return out
}

// encodePNG encodes RGB pixels as an opaque PNG.
func encodePNG(p pixels) ([]byte, error) {
	img := image.NewNRGBA(image.Rect(0, 0, p.w, p.h))
	for i := 0; i < p.w*p.h; i++ {
		img.Pix[i*4], img.Pix[i*4+1], img.Pix[i*4+2], img.Pix[i*4+3] = p.data[i*3], p.data[i*3+1], p.data[i*3+2], 255
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, fmt.Errorf("diffusion: encoding the png: %w", err)
	}
	return buf.Bytes(), nil
}

// DecodePNG decodes PNG or JPEG bytes into an image, for a caller that wants to
// look at a Result or build a mask. It applies the same size limit as a request.
func DecodePNG(b []byte) (image.Image, error) {
	if _, err := decodeConfig(b); err != nil {
		return nil, err
	}
	img, _, err := image.Decode(bytes.NewReader(b))
	return img, err
}

// EncodeMaskPNG builds a mask from a rectangle: white inside it, black outside.
// It is the smallest useful mask, and what the tests and the README examples
// use; an application with a drawing surface sends its own.
func EncodeMaskPNG(width, height int, repaint image.Rectangle) ([]byte, error) {
	if width <= 0 || height <= 0 || width*height > MaxImagePixels {
		return nil, fmt.Errorf("%w: mask size %dx%d", ErrInvalidRequest, width, height)
	}
	img := image.NewGray(image.Rect(0, 0, width, height))
	for y := repaint.Min.Y; y < repaint.Max.Y && y < height; y++ {
		for x := repaint.Min.X; x < repaint.Max.X && x < width; x++ {
			if x >= 0 && y >= 0 {
				img.SetGray(x, y, color.Gray{Y: 255})
			}
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
