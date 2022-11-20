

download_cellpose_data:
	mkdir -p data/cellpose
	cd data/cellpose;
	kaggle datasets download -d ks2019/sartorius-train-tif;
	unzip sartorius-train-tif.zip


generate_broken_masks:
	PYTHONPATH=. python src/broken_masks/generate_broken_masks.py


evaluate_cellpose:
	PYTHONPATH=. python src/cellpose/evaluate.py
