agg_array=("avg" "")
measure_array=("sojourn" )
format_array=("2d" "3d")
r_hour_array=(3 1)
stride_hour_array=(1 )
input_size_array=(24 )
algo_array=("basic_CNN" "basic_LSTM" "basic_LR" "basic_RF" "basic_SVR")
pred_array=("state" "trans")
search="False"
result="hos_exp_result_1"

for pred in ${pred_array[@]}; do
	for measure in ${measure_array[@]}; do
		for r_hour in ${r_hour_array[@]}; do
			for stride_hour in ${stride_hour_array[@]}; do
				for input_size in ${input_size_array[@]}; do
					for format in ${format_array[@]}; do
						for agg in ${agg_array[@]}; do
							python prepare_main.py --exp_name 'HOS' --path './sample_data/hospital.csv' --agg "avg" --measure $measure --fv_format $format --start "2018-01-01 00:00:00" --end "2018-12-31 23:59:59" --range "0,0,0,"$r_hour",0,0" --stride "0,0,0,"$stride_hour",0,0" --input_size $input_size --output_size 1 --edge_threshold 100 --node_threshold 100
							python train_main.py --search 'True' --prediction $pred --exp_id "HOS_avg_"$measure"_2d_2018-01-01-2018-01-01_range0,0,0,"$r_hour",0,0_stride0,0,0,"$stride_hour",0,0_input"$input_size"_output1_100_100" --result $result

							if [ $format = "2d" ]; then
								for algo in ${algo_array[@]}; do
									python train_main.py --search 'False' --fv_format '2d' --prediction $pred --exp_id "HOS_avg_"$measure"_2d_2018-01-01-2018-01-01_range0,0,0,"$r_hour",0,0_stride0,0,0,"$stride_hour",0,0_input"$input_size"_output1_100_100" --algo $algo --result $result
								done;
							else
								python train_main.py --search 'False' --fv_format '3d' --prediction $pred --exp_id "HOS_avg_"$measure"_3d_2018-01-01-2018-01-01_range0,0,0,"$r_hour",0,0_stride0,0,0,"$stride_hour",0,0_input"$input_size"_output1_100_100" --algo 'basic_LRCN' --result $result
							fi
						done;
					done;
				done;
			done;
		done;
	done;
done;