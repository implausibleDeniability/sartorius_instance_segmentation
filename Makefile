## this section contains data with launching cellpose

# download cellpose train/test data
download_cellpose_data:
	mkdir -p data/cellpose
	cd data/cellpose; \
	kaggle datasets download -d ks2019/sartorius-train-tif; \
	unzip sartorius-train-tif.zip

# generates images with broken masks
generate_broken_masks:
	PYTHONPATH=. python src/broken_masks/generate_broken_masks.py

# calculates map score for evaluation
evaluate_cellpose:
	PYTHONPATH=. python src/cellpose/evaluate.py


# generates prediction / target masks for cellpose
generate_masks_cellpose:
    PYTHONPATH=. python src/cellpose/generate_images_for_report.py