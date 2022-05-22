from time import sleep
import pandas as pd
import numpy as np
import resnet2D
import matplotlib.pyplot as plt

def get_class_weights(sleep_disorder):
    # Class weight setting
    # Get class weight for the healthy and specific sleep disorder represented by 'sleep_disorder'
    # --> sleep_disorder can be among 'OSA', 'Insomnia', 'COMISA'

    # load 'Final_sub_ID_grouped.csv'
    df_grouped_class = pd.read_csv('./sub_info/Final_sub_ID_grouped.csv')

    num_healthy = df_grouped_class['Healthy'].dropna().size
    num_disorder = df_grouped_class[sleep_disorder].dropna().size
    total=num_healthy+num_disorder

    # Get class weights
    weight_for_0 = (1 / num_disorder) * (total / 2.0) # sleep disorder
    weight_for_1 = (1 / num_healthy) * (total / 2.0) # Healthy

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    return class_weight

def get_model(model_dim=2, no_layer=1, input_shape=(1,5,660), num_class=2, kernel_size=(2,2)):
    # no_layer --> 1=18, 2=34, 3=50, 4=101, 5=152
    # num_class --> if 1, regression model
    # input_shape --> shape of numpy input in 3D format (nb_channels, nb_rows, nb_columns)
    # -- input_shape는 내장 알고리즘에 의해서 (5,660,1)로 변환됨
    # -- train_data는 (5,660,1) 형식으로 생성되어 있음
    print("now kernel size: {}".format(kernel_size))
    if model_dim == 2:  # 2D
        from resnet2D import ResnetBuilder
        if no_layer==1:
            model = ResnetBuilder.build_resnet_18(input_shape, num_class, kernel_size)
        elif no_layer==2:
            model = ResnetBuilder.build_resnet_34(input_shape, num_class, kernel_size)
        elif no_layer==3:
            model = ResnetBuilder.build_resnet_50(input_shape, num_class, kernel_size)
        elif no_layer==4:
            model = ResnetBuilder.build_resnet_101(input_shape, num_class, kernel_size)
        elif no_layer==5:
            model = ResnetBuilder.build_resnet_152(input_shape, num_class, kernel_size)

    
    model.summary()
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                # loss = 'categorical_crossentropy',
                metrics=['accuracy'])
    

    return model

def plot_acc_loss(H, path_save, epochs, str_kenel_size):
    # input: output from model.fit()
    plt.style.use("ggplot")
    plt.figure(figsize=(20,10))

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.title("Training and Validation Loss_%s" % str_kenel_size, fontsize=20)
    plt.xlabel("Epoch #", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)

    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training and Validation Accuracy_%s" % str_kenel_size, fontsize=20)
    plt.xlabel("Epoch #", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)


    plt.savefig(path_save)
    plt.close()
    # plt.savefig(args["output"])


