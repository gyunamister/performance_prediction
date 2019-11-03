agg_array=("avg" "")
measure_array=("processing" "")
format_array=("2d" "3d")
r_hour_array=(6 )
stride_hour_array=(6 )
input_size_array=(24 )
algo_array=("basic_CNN" "basic_LSTM" "basic_LR" "basic_RF" "basic_SVR")
pred_array=("state" "trans")
search="True"
edge_threshold="100"
node_threshold="100"
result="BPIC12_exp_result_1"
horizon_array=(1 )
for horizon in ${horizon_array[@]}; do
	for pred in ${pred_array[@]}; do
		for measure in ${measure_array[@]}; do
			for r_hour in ${r_hour_array[@]}; do
				for stride_hour in ${stride_hour_array[@]}; do
					for input_size in ${input_size_array[@]}; do
						for format in ${format_array[@]}; do
							for agg in ${agg_array[@]}; do
								python prepare_main.py --exp_name 'BPIC12' --path './sample_data/BPIC2012.csv' --agg "avg" --measure $measure --fv_format $format --start "2011-10-01 00:00:00" --end "2012-03-14 23:59:59" --range "0,0,0,"$r_hour",0,0" --stride "0,0,0,"$stride_hour",0,0" --input_size $input_size --output_size 1 --edge_threshold $edge_threshold --node_threshold $node_threshold --horizon $horizon
								python train_main.py --search 'True' --prediction $pred --exp_id "BPIC12_avg_"$measure"_2d_2011-10-01-2011-10-01_range0,0,0,"$r_hour",0,0_stride0,0,0,"$stride_hour",0,0_input"$input_size"_output1_"$edge_threshold"_"$node_threshold"_"$horizon"" --result $result
								if [ $format = "2d" ]; then
									for algo in ${algo_array[@]}; do
										python train_main.py --search 'False' --fv_format '2d' --prediction $pred --exp_id "BPIC12_avg_"$measure"_2d_2011-10-01-2011-10-01_range0,0,0,"$r_hour",0,0_stride0,0,0,"$stride_hour",0,0_input"$input_size"_output1_"$edge_threshold"_"$node_threshold"_"$horizon"" --algo $algo --result $result
									done;
								else
									python train_main.py --search 'False' --fv_format '3d' --prediction $pred --exp_id "BPIC12_avg_"$measure"_3d_2011-10-01-2011-10-01_range0,0,0,"$r_hour",0,0_stride0,0,0,"$stride_hour",0,0_input"$input_size"_output1_"$edge_threshold"_"$node_threshold"_"$horizon"" --algo 'basic_LRCN' --result $result
								fi
							done;
						done;
					done;
				done;
			done;
		done;
	done;
done;