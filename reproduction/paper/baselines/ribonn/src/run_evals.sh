
python3 -m src.evaluate_predictions --pred_txt results/HEK293/prediction_output_cv.txt --gt_tsv HEK293/test_mean_te.tsv --gt_col TE_HEK293 --out_dir results/HEK293/plots
python3 -m src.evaluate_predictions --pred_txt results/LCL/prediction_output_cv.txt --gt_tsv LCL/test_mean_te.tsv --gt_col TE_LCL --out_dir results/LCL/plots
python3 -m src.evaluate_predictions --pred_txt results/RPE/prediction_output_cv.txt --gt_tsv RPE/test_mean_te.tsv --gt_col TE_RPE-1 --out_dir results/RPE/plots
python3 -m src.evaluate_predictions --pred_txt results/IPSC/prediction_output_cv.txt --gt_tsv IPSC/test_mean_te.tsv --gt_col mean_te --out_dir results/IPSC/plots
