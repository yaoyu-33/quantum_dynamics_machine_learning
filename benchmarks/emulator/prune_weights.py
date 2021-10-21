import emulator.evaluation
import emulator.data
import emulator.model
import os
import argparse
import global_config
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

if __name__ == '__main__':
    """Train and evaluate."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--datasets_path', type=str,
        help='Path to the datasets directory')
    args = parser.parse_args()

    if not args.datasets_path:
        raise Exception('You must specify datasets path.')

    conf_gru = global_config.Config(
        datasets_path=os.path.realpath(
                os.path.expanduser(args.datasets_path)),
        model_name="demo-gru",
    )

    # easy_case = emulator.data._retrieve_validation_data(args.datasets_path + "test/full_test_E0_0.0to9.0_BH_0.0to14.0/X0_15.1_S0_1.7_E0_6.4_BH_7.4_BW_9.4.txt")
    # inter_case = emulator.data._retrieve_validation_data(args.datasets_path + "test/full_test_E0_0.0to9.0_BH_0.0to14.0/triple.txt")
    # comp_case = emulator.data._retrieve_validation_data(args.datasets_path + "test/full_test_E0_0.0to9.0_BH_0.0to14.0/random.txt")

    results = pd.DataFrame(columns = ['name','gru','gru_prune_0.00001','gru_prune_0.0001','gru_prune_0.001','gru_prune_0.01'])
    testset = os.listdir(args.datasets_path + "test/full_test_E0_0.0to9.0_BH_0.0to14.0/")

    model1 = emulator.model.RNNModel(conf_gru)
    model1.load_weights("../models/demo-gru/ckpts/final_step")
    model1.build((None, conf_gru.input_frames, conf_gru.window_size, conf_gru.input_channels))
    
    # print(model1.weights)
    # easy_result = emulator.evaluation.get_result(model1,conf_gru,easy_case)
    # inter_result = emulator.evaluation.get_result(model1,conf_gru,inter_case)
    # comp_result = emulator.evaluation.get_result(model1,conf_gru,comp_case)
    # easy_corr1 = emulator.evaluation.get_avg_corr(conf_gru,easy_case,easy_result)
    # inter_corr1 = emulator.evaluation.get_avg_corr(conf_gru,inter_case,inter_result)
    # comp_corr1 = emulator.evaluation.get_avg_corr(conf_gru,comp_case,comp_result)
    

    model2 = emulator.model.RNNModel(conf_gru)
    model2.load_weights("../models/demo-gru/ckpts/final_step")
    model2.build((None, conf_gru.input_frames, conf_gru.window_size, conf_gru.input_channels))
    weights = model2.layers[1].get_weights()
    print(model2.layers[1].weights)
    # print(model2.layers[1].weights)
    weights[0] = (abs(weights[0]) > 0.00001) * weights[0]
    model2.layers[1].set_weights(weights)
    # easy_result = emulator.evaluation.get_result(model2,conf_gru,easy_case)
    # inter_result = emulator.evaluation.get_result(model2,conf_gru,inter_case)
    # comp_result = emulator.evaluation.get_result(model2,conf_gru,comp_case)
    # easy_corr2 = emulator.evaluation.get_avg_corr(conf_gru,easy_case,easy_result)
    # inter_corr2 = emulator.evaluation.get_avg_corr(conf_gru,inter_case,inter_result)
    # comp_corr2 = emulator.evaluation.get_avg_corr(conf_gru,comp_case,comp_result)


    model3 = emulator.model.RNNModel(conf_gru)
    model3.load_weights(f"../models/demo-gru/ckpts/final_step")
    model3.build((None, conf_gru.input_frames, conf_gru.window_size, conf_gru.input_channels))
    weights = model3.layers[1].get_weights()
    weights[0] = (abs(weights[0]) > 0.0001) * weights[0]
    model3.layers[1].set_weights(weights)
    # easy_result = emulator.evaluation.get_result(model3,conf_gru,easy_case)
    # inter_result = emulator.evaluation.get_result(model3,conf_gru,inter_case)
    # comp_result = emulator.evaluation.get_result(model3,conf_gru,comp_case)
    # easy_corr3 = emulator.evaluation.get_avg_corr(conf_gru,easy_case,easy_result)
    # inter_corr3 = emulator.evaluation.get_avg_corr(conf_gru,inter_case,inter_result)
    # comp_corr3 = emulator.evaluation.get_avg_corr(conf_gru,comp_case,comp_result)


    model4 = emulator.model.RNNModel(conf_gru)
    model4.load_weights(f"../models/demo-gru/ckpts/final_step")
    model4.build((None, conf_gru.input_frames, conf_gru.window_size, conf_gru.input_channels))
    weights = model4.layers[1].get_weights()
    weights[0] = (abs(weights[0]) > 0.001) * weights[0]
    model4.layers[1].set_weights(weights)
    # easy_result = emulator.evaluation.get_result(model4,conf_gru,easy_case)
    # inter_result = emulator.evaluation.get_result(model4,conf_gru,inter_case)
    # comp_result = emulator.evaluation.get_result(model4,conf_gru,comp_case)
    # easy_corr4 = emulator.evaluation.get_avg_corr(conf_gru,easy_case,easy_result)
    # inter_corr4 = emulator.evaluation.get_avg_corr(conf_gru,inter_case,inter_result)
    # comp_corr4 = emulator.evaluation.get_avg_corr(conf_gru,comp_case,comp_result)


    model5 = emulator.model.RNNModel(conf_gru)
    model5.load_weights(f"../models/demo-gru/ckpts/final_step")
    model5.build((None, conf_gru.input_frames, conf_gru.window_size, conf_gru.input_channels))
    weights = model5.layers[1].get_weights()
    weights[0] = (abs(weights[0]) > 0.01) * weights[0]
    model5.layers[1].set_weights(weights)
    # easy_result = emulator.evaluation.get_result(model5,conf_gru,easy_case)
    # inter_result = emulator.evaluation.get_result(model5,conf_gru,inter_case)
    # comp_result = emulator.evaluation.get_result(model5,conf_gru,comp_case)
    # easy_corr5 = emulator.evaluation.get_avg_corr(conf_gru,easy_case,easy_result)
    # inter_corr5 = emulator.evaluation.get_avg_corr(conf_gru,inter_case,inter_result)
    # comp_corr5 = emulator.evaluation.get_avg_corr(conf_gru,comp_case,comp_result)


    # fig1 = plt.figure(figsize = (12,8))
    # plt.plot(easy_corr1, label = 'gru')
    # plt.plot(easy_corr2, label = 'gru_prune_0.00001')
    # plt.plot(easy_corr3, label = 'gru_prune_0.0001')
    # plt.plot(easy_corr4, label = 'gru_prune_0.001')
    # plt.plot(easy_corr5, label = 'gru_prune_0.01')
    # plt.xlabel('Time')
    # plt.ylabel('Correlation')
    # plt.title('Cumulative correlation (EASY)')
    # plt.legend()
    # plt.savefig("graphs/easy_corr_pruned.png")


    # fig2 = plt.figure(figsize = (12,8))
    # plt.plot(inter_corr1, label = 'gru')
    # plt.plot(inter_corr2, label = 'gru_prune_0.00001')
    # plt.plot(inter_corr3, label = 'gru_prune_0.0001')
    # plt.plot(inter_corr4, label = 'gru_prune_0.001')
    # plt.plot(inter_corr5, label = 'gru_prune_0.01')
    # plt.xlabel('Time')
    # plt.ylabel('Correlation')
    # plt.title('Cumulative correlation (INTERMEDIATE)')
    # plt.legend()
    # plt.savefig("graphs/inter_corr_pruned.png")


    # fig3 = plt.figure(figsize = (12,8))
    # plt.plot(comp_corr1, label = 'gru')
    # plt.plot(comp_corr2, label = 'gru_prune_0.00001')
    # plt.plot(comp_corr3, label = 'gru_prune_0.0001')
    # plt.plot(comp_corr4, label = 'gru_prune_0.001')
    # plt.plot(comp_corr5, label = 'gru_prune_0.01')
    # plt.xlabel('Time')
    # plt.ylabel('Correlation')
    # plt.title('Cumulative correlation (COMPLEX)')
    # plt.legend()
    # plt.savefig("graphs/comp_corr_pruned.png")

    for test in testset:
        print(test)
        if test[-4:] == '.txt':
            pre = args.datasets_path + "test/full_test_E0_0.0to9.0_BH_0.0to14.0/"
            case = emulator.data._retrieve_validation_data(pre+test)
            m1_result = emulator.evaluation.get_result(model1,conf_gru,case)
            m1_corr = np.average(emulator.evaluation.get_avg_corr(conf_gru,case,m1_result))
            m2_result = emulator.evaluation.get_result(model2,conf_gru,case)
            m2_corr = np.average(emulator.evaluation.get_avg_corr(conf_gru,case,m2_result))
            m3_result = emulator.evaluation.get_result(model3,conf_gru,case)
            m3_corr = np.average(emulator.evaluation.get_avg_corr(conf_gru,case,m3_result))
            m4_result = emulator.evaluation.get_result(model4,conf_gru,case)
            m4_corr = np.average(emulator.evaluation.get_avg_corr(conf_gru,case,m4_result))
            m5_result = emulator.evaluation.get_result(model5,conf_gru,case)
            m5_corr = np.average(emulator.evaluation.get_avg_corr(conf_gru,case,m5_result))

            results = results.append({'name':test,'gru':m1_corr,'gru_prune_0.00001':m2_corr,'gru_prune_0.0001':m3_corr,'gru_prune_0.001':m4_corr,'gru_prune_0.01':m5_corr}, ignore_index=True)
            print(results)
        else:
            continue
    
    results.to_csv('prune_dense_results.csv')