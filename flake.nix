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
    nixpkgs.url = "github:NixOS/nixpkgs/a438c431ba249e81f2bdb3dedd6b79492d2b7238";
    # nixpkgs.url = "nixpkgs/nixos-23.11";
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
        # packages = {
        #   openai-triton-cuda = pkgs.openai-triton-cuda.overrideAttrs ( old: rec {
        #     version = "2.4.9";
        #     src = pkgs.fetchFromGitHub {
        #       owner = "doctest";
        #       repo = "doctest";
        #       rev = "v${version}";
        #       sha256 = "sha256-ugmkeX2PN4xzxAZpWgswl4zd2u125Q/ADSKzqTfnd94=";
        #     };
        #     patches = [];
        #   });
        # }
        devShell = pkgs.mkShell {
          buildInputs = (with pkgs; [
            python3
          ]) ++ (with pkgs.python3Packages; [
            openai-triton-cuda
            numpy
            # scipy
            pytorch-bin
          ]);
        };
      }
  );
}
