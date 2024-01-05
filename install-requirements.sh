# pin these versions for pallas support
# we don't use a requirements file as they don't support these specific flags
# TODO: use something like setup.py to do this instead
pip install --no-deps -IU --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly==2.1.0.post20231216005823
pip install -IU --pre jax==0.4.24.dev20240104 -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
pip install -IU --pre "jaxlib[cuda12_pip]==0.4.24.dev20240103" -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_cuda12_releases.html
pip install --no-deps absl-py 'jax-triton @ git+https://github.com/jax-ml/jax-triton.git@7778c47c0a27c0988c914dce640dec61e44bbe8c'

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/patrick-kidger/equinox.git@dev einops huggingface_hub transformers
