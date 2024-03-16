{
  nixConfig = {
    bash-prompt-prefix = "(triton) ";
  };

  # Nixpkgs / NixOS version to use.
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-23.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }: flake-utils.lib.eachSystem [ "x86_64-linux" ] (system: 
    let 
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in 
      {
        devShell = pkgs.mkShell {
          buildInputs = (with pkgs; [
            python3
          ]) ++ (with pkgs.python3Packages; [
            numpy
            scipy
            torch
          ]);
        };
      }
  );
}