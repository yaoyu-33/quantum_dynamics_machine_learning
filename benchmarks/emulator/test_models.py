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
import logging

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
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

    conf_reg_l1 = global_config.Config(
        datasets_path=os.path.realpath(
                os.path.expanduser(args.datasets_path)),
        model_name="demo-gru-reg-l1",
    )

    conf_reg_l2 = global_config.Config(
        datasets_path=os.path.realpath(
                os.path.expanduser(args.datasets_path)),
        model_name="demo-gru-reg-l2",
    )

    conf_reg_l2_high = global_config.Config(
        datasets_path=os.path.realpath(
                os.path.expanduser(args.datasets_path)),
        model_name="demo-gru-reg-l2-high",
    )

    # easy_case = emulator.data._retrieve_validation_data(args.datasets_path + "test/full_test_E0_0.0to9.0_BH_0.0to14.0/X0_15.1_S0_1.7_E0_6.4_BH_7.4_BW_9.4.txt")
    # inter_case = emulator.data._retrieve_validation_data(args.datasets_path + "test/full_test_E0_0.0to9.0_BH_0.0to14.0/triple.txt")
    # comp_case = emulator.data._retrieve_validation_data(args.datasets_path + "test/full_test_E0_0.0to9.0_BH_0.0to14.0/random.txt")


    results = pd.DataFrame(columns = ['name','gru','gru_reg_l1','gru_reg_l2','gru_reg_l2_high'])
    testset = os.listdir(args.datasets_path + "test/full_test_E0_0.0to9.0_BH_0.0to14.0/")

    model1 = emulator.model.RNNModel(conf_gru)
    model1.load_weights(f"../models/{conf_gru.model_name}/ckpts/final_step")
    
    # easy_result = emulator.evaluation.get_result(model1,conf_gru,easy_case)
    # inter_result = emulator.evaluation.get_result(model1,conf_gru,inter_case)
    # comp_result = emulator.evaluation.get_result(model1,conf_gru,comp_case)
    # easy_corr1 = emulator.evaluation.get_avg_corr(conf_gru,easy_case,easy_result)
    # inter_corr1 = emulator.evaluation.get_avg_corr(conf_gru,inter_case,inter_result)
    # comp_corr1 = emulator.evaluation.get_avg_corr(conf_gru,comp_case,comp_result)
    

    model2 = emulator.model.RNNModel(conf_reg_l1, reg=True, type=tf.keras.regularizers.L1(0.01))
    model2.load_weights(f"../models/{conf_reg_l1.model_name}/ckpts/final_step")
    # easy_result = emulator.evaluation.get_result(model2,conf_reg_l1,easy_case)
    # inter_result = emulator.evaluation.get_result(model2,conf_reg_l1,inter_case)
    # comp_result = emulator.evaluation.get_result(model2,conf_reg_l1,comp_case)
    # easy_corr2 = emulator.evaluation.get_avg_corr(conf_reg_l1,easy_case,easy_result)
    # inter_corr2 = emulator.evaluation.get_avg_corr(conf_reg_l1,inter_case,inter_result)
    # comp_corr2 = emulator.evaluation.get_avg_corr(conf_reg_l1,comp_case,comp_result)


    model3 = emulator.model.RNNModel(conf_reg_l2, reg=True, type=tf.keras.regularizers.L2(0.01))
    model3.load_weights(f"../models/{conf_reg_l2.model_name}/ckpts/final_step")
    # easy_result = emulator.evaluation.get_result(model3,conf_reg_l2,easy_case)
    # inter_result = emulator.evaluation.get_result(model3,conf_reg_l2,inter_case)
    # comp_result = emulator.evaluation.get_result(model3,conf_reg_l2,comp_case)
    # easy_corr3 = emulator.evaluation.get_avg_corr(conf_reg_l2,easy_case,easy_result)
    # inter_corr3 = emulator.evaluation.get_avg_corr(conf_reg_l2,inter_case,inter_result)
    # comp_corr3 = emulator.evaluation.get_avg_corr(conf_reg_l2,comp_case,comp_result)


    model4 = emulator.model.RNNModel(conf_reg_l2_high, reg=True, type=tf.keras.regularizers.L2(0.1))
    model4.load_weights(f"../models/{conf_reg_l2_high.model_name}/ckpts/final_step")
    # easy_result = emulator.evaluation.get_result(model4,conf_reg_l2_high,easy_case)
    # inter_result = emulator.evaluation.get_result(model4,conf_reg_l2_high,inter_case)
    # comp_result = emulator.evaluation.get_result(model4,conf_reg_l2_high,comp_case)
    # easy_corr4 = emulator.evaluation.get_avg_corr(conf_reg_l2_high,easy_case,easy_result)
    # inter_corr4 = emulator.evaluation.get_avg_corr(conf_reg_l2_high,inter_case,inter_result)
    # comp_corr4 = emulator.evaluation.get_avg_corr(conf_reg_l2_high,comp_case,comp_result)

    # fig1 = plt.figure(figsize = (12,8))
    # plt.plot(easy_corr1, label = 'gru')
    # plt.plot(easy_corr2, label = 'gru_reg_l1')
    # plt.plot(easy_corr3, label = 'gru_reg_l2')
    # plt.plot(easy_corr4, label = 'gru_reg_l2_high')
    # plt.xlabel('Time')
    # plt.ylabel('Correlation')
    # plt.title('Cumulative correlation (EASY)')
    # plt.legend()
    # plt.savefig("graphs/easy_corr_reg.png")


    # fig2 = plt.figure(figsize = (12,8))
    # plt.plot(inter_corr1, label = 'gru')
    # plt.plot(inter_corr2, label = 'gru_reg_l1')
    # plt.plot(inter_corr3, label = 'gru_reg_l2')
    # plt.plot(inter_corr4, label = 'gru_reg_l2_high')
    # plt.xlabel('Time')
    # plt.ylabel('Correlation')
    # plt.title('Cumulative correlation (INTERMEDIATE)')
    # plt.legend()
    # plt.savefig("graphs/inter_corr_reg.png")


    # fig3 = plt.figure(figsize = (12,8))
    # plt.plot(comp_corr1, label = 'gru')
    # plt.plot(comp_corr2, label = 'gru_reg_l1')
    # plt.plot(comp_corr3, label = 'gru_reg_l2')
    # plt.plot(comp_corr4, label = 'gru_reg_l2_high')
    # plt.xlabel('Time')
    # plt.ylabel('Correlation')
    # plt.title('Cumulative correlation (COMPLEX)')
    # plt.legend()
    # plt.savefig("graphs/comp_corr_reg.png")


    for test in testset:
        logger.debug(test)
        if test[-4:] == '.txt':
            pre = args.datasets_path + "test/full_test_E0_0.0to9.0_BH_0.0to14.0/"
            case = emulator.data._retrieve_validation_data(pre+test)
            m1_result = emulator.evaluation.get_result(model1,conf_gru,case)
            m1_corr = np.average(emulator.evaluation.get_avg_corr(conf_gru,case,m1_result))
            m2_result = emulator.evaluation.get_result(model2,conf_reg_l1,case)
            m2_corr = np.average(emulator.evaluation.get_avg_corr(conf_reg_l1,case,m2_result))
            m3_result = emulator.evaluation.get_result(model3,conf_reg_l2,case)
            m3_corr = np.average(emulator.evaluation.get_avg_corr(conf_reg_l2,case,m3_result))
            m4_result = emulator.evaluation.get_result(model4,conf_reg_l2_high,case)
            m4_corr = np.average(emulator.evaluation.get_avg_corr(conf_reg_l2_high,case,m4_result))

            results = results.append({'name':test,'gru':m1_corr,'gru_reg_l1':m2_corr,'gru_reg_l2':m3_corr,'gru_reg_l2_high':m4_corr}, ignore_index=True)
            logger.debug(results)
        else:
            continue

    results.to_csv('reg_results.csv')

        
        




