{
  description = "A basic flake providing a devShell";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { nixpkgs, ... }: {
    devShells = builtins.mapAttrs (system: pkgs: {
      default = pkgs.mkShell {
        packages = with pkgs; [
          (python3.withPackages (
            ps: with ps; [
              notebook

              numpy
              matplotlib
            ]
          ))

          jupyter

          basedpyright
          ruff
        ];
      };
    }) nixpkgs.legacyPackages;
  };
}
