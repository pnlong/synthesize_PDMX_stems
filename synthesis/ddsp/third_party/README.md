# Placeholder: clone DDSP-Piano here for the isolated TF worker.
#
#   git clone https://github.com/lrenault/ddsp-piano.git synthesis/ddsp/third_party/ddsp-piano
#
# Default checkpoint (bundled in upstream):
#   ddsp_piano/model_weights/dafx22
# Gin config:
#   ddsp_piano/configs/dafx22.gin
#
# Citation (from upstream README BibTeX — Renault, Mignot, Roebel):
#   @article{renault2023ddsp_piano, JAES 71(9):552–565, Sept 2023}
#   @inproceedings{renault2022diffpiano, DAFx 2022}
# Training data: MAESTRO (Disklavier / International Piano-e-Competition).
# License: Apache-2.0
#
# Spot-listen gate (run after Track C setup):
#   SPDMX_DDSP_PYTHON=$PWD/.venv-ddsp/bin/python \\
#     uv run python -m synthesis.ddsp.spot_listen_piano --out /tmp/ddsp_piano_spot.wav
