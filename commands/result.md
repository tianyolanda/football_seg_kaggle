# define
class_name_list = ['Background', 'Team A', 'Team B', 'Goalkeeper A', 'Goalkeeper B',
                    'Referee', 'Ball', 'Goal Bar', 'Audience', 
                    'Advertisement', 'Ground']
# 1 vgg_unet 
Verifying training dataset
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 90/90 [00:06<00:00, 14.10it/s]
Dataset verified! 
Epoch 1/5
512/512 [==============================] - ETA: 0s - loss: 0.4570 - accuracy: 0.8780     
Epoch 1: saving model to /tmp/vgg_unet_1.00001
512/512 [==============================] - 6833s 13s/step - loss: 0.4570 - accuracy: 0.8780
Epoch 2/5
512/512 [==============================] - ETA: 0s - loss: 0.1187 - accuracy: 0.9653     
Epoch 2: saving model to /tmp/vgg_unet_1.00002
512/512 [==============================] - 2854s 6s/step - loss: 0.1187 - accuracy: 0.9653
Epoch 3/5
512/512 [==============================] - ETA: 0s - loss: 0.0572 - accuracy: 0.9815     
Epoch 3: saving model to /tmp/vgg_unet_1.00003
512/512 [==============================] - 5686s 11s/step - loss: 0.0572 - accuracy: 0.9815
Epoch 4/5
512/512 [==============================] - ETA: 0s - loss: 0.0444 - accuracy: 0.9851   
Epoch 4: saving model to /tmp/vgg_unet_1.00004
512/512 [==============================] - 2881s 6s/step - loss: 0.0444 - accuracy: 0.9851
Epoch 5/5
512/512 [==============================] - ETA: 0s - loss: 0.0241 - accuracy: 0.9914   
Epoch 5: saving model to /tmp/vgg_unet_1.00005
512/512 [==============================] - 3401s 7s/step - loss: 0.0241 - accuracy: 0.9914
1/1 [==============================] - 1s 1s/step
1/1 [==============================] - 1s 689ms/step
1/1 [==============================] - 1s 663ms/step
1/1 [==============================] - 1s 665ms/step
1/1 [==============================] - 1s 831ms/step
1/1 [==============================] - 1s 650ms/step
1/1 [==============================] - 1s 698ms/step
1/1 [==============================] - 1s 718ms/step
1/1 [==============================] - 1s 713ms/step
1/1 [==============================] - 1s 642ms/step
1/1 [==============================] - 1s 686ms/step
10it [00:08,  1.21it/s]
{'frequency_weighted_IU': 0.9453554242657871, 'mean_IU': 0.7400923817445009, 'class_wise_IU': array([0.        , 0.60041987, 0.7788296 , 0.86654006, 0.90695144,
       0.83951856, 0.50864198, 0.74727534, 0.96688631, 0.94487109,
       0.98108195])}
# 2 fpn_8

1/1 [==============================] - 1s 626ms/step
1/1 [==============================] - 1s 638ms/step
1/1 [==============================] - 1s 623ms/step
1/1 [==============================] - 1s 614ms/step
1/1 [==============================] - 1s 644ms/step
1/1 [==============================] - 1s 634ms/step
1/1 [==============================] - 1s 624ms/step
1/1 [==============================] - 1s 630ms/step
1/1 [==============================] - 1s 613ms/step
1/1 [==============================] - 1s 622ms/step
10it [00:08,  1.16it/s]
{'frequency_weighted_IU': 0.9173012782023601, 'mean_IU': 0.7118064926381612, 'class_wise_IU': array([0.        , 0.61106586, 0.76185012, 0.83380102, 0.84638709,
       0.78620752, 0.46027201, 0.70702532, 0.93779907, 0.92573844,
       0.95972497])}

class_name_list = ['Background', 'Team A', 'Team B', 'Goalkeeper A', 'Goalkeeper B',
                    'Referee', 'Ball', 'Goal Bar', 'Audience', 
                    'Advertisement', 'Ground']


# 3 vgg_unet 416， 

1/1 [==============================] - 1s 1s/step
1/1 [==============================] - 1s 1s/step
1/1 [==============================] - 1s 1s/step
1/1 [==============================] - 1s 1s/step
1/1 [==============================] - 1s 1s/step
1/1 [==============================] - 1s 1s/step
1/1 [==============================] - 1s 1s/step
1/1 [==============================] - 1s 1s/step
1/1 [==============================] - 1s 1s/step
1/1 [==============================] - 1s 1s/step
10it [00:15,  1.58s/it]
{'frequency_weighted_IU': 0.9524968837035814, 'mean_IU': 0.7972437415387188, 'class_wise_IU': array([0.        , 0.71257318, 0.83964212, 0.88772265, 0.89779932,
       0.88771359, 0.79835014, 0.83916254, 0.96887721, 0.95477425,
       0.98306615])}


# 4 segnet

finish validation, and saved in  dataset2/test_result/resnet50_segnet_416_608_dataset2_ori/
1/1 [==============================] - 1s 665ms/step
1/1 [==============================] - 1s 652ms/step
1/1 [==============================] - 1s 661ms/step
1/1 [==============================] - 1s 658ms/step
1/1 [==============================] - 1s 664ms/step
1/1 [==============================] - 1s 646ms/step
1/1 [==============================] - 1s 654ms/step
1/1 [==============================] - 1s 657ms/step
1/1 [==============================] - 1s 660ms/step
1/1 [==============================] - 1s 666ms/step
10it [00:07,  1.27it/s]
{'frequency_weighted_IU': 0.9276700262281972, 'mean_IU': 0.6900688084623208, 'class_wise_IU': array([0.        , 0.37764877, 0.72494013, 0.85568293, 0.87465905,
       0.72001621, 0.43447581, 0.75610956, 0.95752268, 0.92680839,
       0.96289337]), 'class_wise_pixels_norm': array([0.        , 0.00534698, 0.00412608, 0.08637241, 0.09174152,
       0.00375443, 0.00156092, 0.01862348, 0.27479441, 0.18550418,
       0.32817561])}
       


0  in val set
/Users/yutian/Desktop/kaggle/football-semantic-segmentation/only_images/Frame 1  (1).jpg /Users/yutian/Desktop/kaggle/football-semantic-segmentation/old_labels_in_class_v2/Frame 1  (1).jpg___fuse___label_in_class_v2.png
10  in val set
/Users/yutian/Desktop/kaggle/football-semantic-segmentation/only_images/Frame 1  (18).jpg /Users/yutian/Desktop/kaggle/football-semantic-segmentation/old_labels_in_class_v2/Frame 1  (18).jpg___fuse___label_in_class_v2.png
20  in val set
/Users/yutian/Desktop/kaggle/football-semantic-segmentation/only_images/Frame 1  (27).jpg /Users/yutian/Desktop/kaggle/football-semantic-segmentation/old_labels_in_class_v2/Frame 1  (27).jpg___fuse___label_in_class_v2.png
30  in val set
/Users/yutian/Desktop/kaggle/football-semantic-segmentation/only_images/Frame 1  (36).jpg /Users/yutian/Desktop/kaggle/football-semantic-segmentation/old_labels_in_class_v2/Frame 1  (36).jpg___fuse___label_in_class_v2.png
40  in val set
/Users/yutian/Desktop/kaggle/football-semantic-segmentation/only_images/Frame 1  (45).jpg /Users/yutian/Desktop/kaggle/football-semantic-segmentation/old_labels_in_class_v2/Frame 1  (45).jpg___fuse___label_in_class_v2.png
50  in val set
/Users/yutian/Desktop/kaggle/football-semantic-segmentation/only_images/Frame 1  (54).jpg /Users/yutian/Desktop/kaggle/football-semantic-segmentation/old_labels_in_class_v2/Frame 1  (54).jpg___fuse___label_in_class_v2.png
60  in val set
/Users/yutian/Desktop/kaggle/football-semantic-segmentation/only_images/Frame 1  (63).jpg /Users/yutian/Desktop/kaggle/football-semantic-segmentation/old_labels_in_class_v2/Frame 1  (63).jpg___fuse___label_in_class_v2.png
70  in val set
/Users/yutian/Desktop/kaggle/football-semantic-segmentation/only_images/Frame 1  (72).jpg /Users/yutian/Desktop/kaggle/football-semantic-segmentation/old_labels_in_class_v2/Frame 1  (72).jpg___fuse___label_in_class_v2.png
80  in val set
/Users/yutian/Desktop/kaggle/football-semantic-segmentation/only_images/Frame 1  (81).jpg /Users/yutian/Desktop/kaggle/football-semantic-segmentation/old_labels_in_class_v2/Frame 1  (81).jpg___fuse___label_in_class_v2.png
90  in val set
/Users/yutian/Desktop/kaggle/football-semantic-segmentation/only_images/Frame 1  (90).jpg /Users/yutian/Desktop/kaggle/football-semantic-segmentation/old_labels_in_class_v2/Frame 1  (90).jpg___fuse___label_in_class_v2.png
All done!
(py37) localhost:footb