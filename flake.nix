{
      description = "Python 3.11 development environment";
      outputs = { self, nixpkgs }:
      let
        system = "x86_64-linux";
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in {
        devShells.${system}.default = (pkgs.buildFHSEnv {
          name = "nvidia-fuck-you";
          targetPkgs = pkgs: (with pkgs; [
            linuxPackages.nvidia_x11
            libGLU libGL
            xorg.libXi xorg.libXmu freeglut
            xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
            ncurses5 stdenv.cc binutils
            ffmpeg

            # Micromamba does the real legwork
            micromamba
          ]);

          profile = ''
              export LD_LIBRARY_PATH="${pkgs.linuxPackages.nvidia_x11}/lib"
              export CUDA_PATH="${pkgs.cudatoolkit}"
              export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
              export EXTRA_CCFLAGS="-I/usr/include"
              
              # Create and activate micromamba environment if it doesn't exist
              if ! micromamba env list | grep -q "my-environment"; then
                micromamba env create \
                  -n my-environment \
                  anaconda::cudatoolkit \
                  anaconda::cudnn \
                  "anaconda::pytorch=*=*cuda*"
              fi
              
              # Activate the environment
              eval "$(micromamba shell hook --shell=bash)"
              micromamba activate my-environment
          '';

          runScript = "bash";
        }).env;
      };
    }