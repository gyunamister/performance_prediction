agg_array=("avg" "")
measure_array=("sojourn" "")
format_array=("2d" "3d")
r_hour_array=(300 )
stride_hour_array=(24)
input_size_array=(6 )
algo_array=("basic_CNN" "basic_LSTM" "basic_LR" "basic_RF" "basic_SVR")
pred_array=("state" "trans")
result="HD_exp_result_1"

for pred in ${pred_array[@]}; do
	for measure in ${measure_array[@]}; do
		for r_hour in ${r_hour_array[@]}; do
			for stride_hour in ${stride_hour_array[@]}; do
				for input_size in ${input_size_array[@]}; do
					for format in ${format_array[@]}; do
						for agg in ${agg_array[@]}; do
							python prepare_main.py --exp_name 'HD' --path './sample_data/helpdesk2.csv' --agg 'avg' --measure 'sojourn' --fv_format $format --start "2011-01-01 00:00:00" --end "2012-06-30 23:59:59" --range "0,0,0,"$r_hour",0,0" --stride "0,0,0,"$stride_hour",0,0" --input_size $input_size --output_size 1 --edge_threshold 0 --node_threshold 100 --horizon 1
							python train_main.py --search 'True' --prediction $pred --exp_id "HD_avg_sojourn_2d_2011-01-01-2011-01-01_range0,0,0,"$r_hour",0,0_stride0,0,0,"$stride_hour",0,0_input"$input_size"_output1_0_100" --result $result
							if [ $format = "2d" ]; then
								for algo in ${algo_array[@]}; do
									python train_main.py --search 'False' --fv_format '2d' --prediction $pred --exp_id "HD_avg_sojourn_2d_2011-01-01-2011-01-01_range0,0,0,"$r_hour",0,0_stride0,0,0,"$stride_hour",0,0_input"$input_size"_output1_0_100" --algo $algo --result $result
								done;
							else
								python train_main.py --search 'False' --fv_format '3d' --prediction $pred --exp_id "HD_avg_sojourn_3d_2011-01-01-2011-01-01_range0,0,0,"$r_hour",0,0_stride0,0,0,"$stride_hour",0,0_input"$input_size"_output1_0_100" --algo 'basic_LRCN'
							fi
						done;
					done;
				done;
			done;
		done;
	done;
done;