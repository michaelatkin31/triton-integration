{
  nixConfig = {
    bash-prompt-prefix = "(triton) ";

    extra-substituters = [
      "https://cuda-maintainers.cachix.org/"
    ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  # Nixpkgs / NixOS version to use.
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }: flake-utils.lib.eachSystem [ "x86_64-linux" ] (system: 
    let 
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          # cudaSupport = true;
        };
      };
    in 
      {
        devShell = pkgs.mkShell {
          buildInputs = (with pkgs; [
            python3
          ]) ++ (with pkgs.python3Packages; [
            openai-triton-cuda
            numpy
            scipy
            torch
          ]);
        };
      }
  );
}
