# Upgrade Guide

## Unreleased

## v0.1.0

First release; nothing to upgrade from.

To use the real engine, build the archives once (`make archives`) and build the
application with `CGO_ENABLED=1 CGO_LDFLAGS="-L<path>/lib" go build -tags sdcpp`.
Run `aru migrate` for the `diffusions` table, and open the policy with
`Config.Policy`: the shipped one denies every action.

