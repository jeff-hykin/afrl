{ main }:
    let
        ikillSnowball = (builtins.import (../packages/ikill.nix));
        ikillOutputs = (ikillSnowball.outputs (ikillSnowball));
        ikillPackage = ikillOutputs.preflakePackage;
    in
        {
            buildInputs = [ ikillPackage ];
        }