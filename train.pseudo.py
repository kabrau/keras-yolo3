#! /usr/bin/env python

import argparse
import os
import numpy as np
import json
from voc import parse_voc_annotation
from yolo import create_yolov3_model, dummy_loss
from generator import BatchGenerator
from utils.utils import normalize, evaluate, makedirs, get_yolo_boxes
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from callbacks import CustomModelCheckpoint, CustomTensorBoard
from utils.multi_gpu_model import multi_gpu_model
import tensorflow as tf
import keras
from keras.models import load_model
import cv2


def PredictToBndBox(filename, raw_height, raw_width, prediction, labels, score_threshold = 0.3):
    img = {'object':[]}
    img['filename'] = filename
    img['width'] = raw_width
    img['height'] = raw_height

    # sort the boxes according to scores
    score = np.array([box.get_score() for box in prediction])
    score_sort = np.argsort(-score)
    boxes = np.array([box for box in prediction])
    boxes = boxes[score_sort]

    for box in boxes:
        if box.score>score_threshold:
            obj = {}
            obj['name'] = labels[box.label]
            obj['xmin'] = box.xmin 
            obj['ymin'] = box.ymin 
            obj['xmax'] = box.xmax 
            obj['ymax'] = box.ymax 
            obj['score'] = box.score
            img['object'] += [obj]

    return img

def calcBestScore(pseudo_images, real_images, labels):
    
    iou_threshold = 0.3
    score_threshold = 0.3

    curves = []

    # scores 0, 0.01, 0.02, ..., 0.99
    for score_threshold in np.arange(0, 1.0, 0.05):
        all_detections  = [[[] for i in range(len(labels))] for j in range(len(pseudo_images))]
        all_annotations  = [[[] for i in range(len(labels))] for j in range(len(pseudo_images))]

        total_annotations = 0

        for i in range(len(pseudo_images)):

            for ilabel in range(len(labels)):
                label = labels[ilabel]
                hasDetection = False
                for box in pseudo_images[i]['object']:
                    if (box['score']>=score_threshold):
                        if (box['name'] == label):
                            all_detections[i][ilabel] += [[box['xmin'],box['ymin'],box['xmax'],box['ymax'],box['score']]]
                            hasDetection = True
                
                if (hasDetection):
                    for box in real_images[i]['object']:
                        if (box['name'] == label):
                            all_annotations[i][ilabel] += [[box['xmin'],box['ymin'],box['xmax'],box['ymax']]]

        # compute mAP by comparing all detections and all annotations
        average_precisions = {}
        
        for label in range(len(labels)):
            false_positives = np.zeros((0,))
            true_positives  = np.zeros((0,))
            scores          = np.zeros((0,))
            num_annotations = 0.0

            for i in range(len(pseudo_images)):
                detections           = all_detections[i][label] 
                annotations          = all_annotations[i][label] 
                num_annotations     += len(annotations)
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4]) 

                    if len(annotations) == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)
                        continue

                    a = np.array(np.expand_dims(d, axis=0))
                    b = np.array(annotations)

                    overlaps            = compute_overlap(a, b)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap         = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives  = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)        

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            # sort by score
            indices         = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives  = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives  = np.cumsum(true_positives)

            # compute recall and precision
            recall    = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision  = compute_ap(recall, precision)  
            average_precisions[label] = average_precision

            total_annotations += num_annotations

        obj = {}
        obj['score'] = score_threshold
        obj['total'] = total_annotations
        obj['average'] = average_precisions
        curves += [obj]
    
    for avg in curves:
        #for label, average_precision in average_precisions.items():
        #    print(self.labels[label], '{:.4f}'.format(average_precision))
        print('Score {:.2f}'.format(avg['score']),'Total {}'.format(avg['total']),'mAP: {:.4f}'.format(sum(avg['average'].values()) / len(avg['average']))) 
    return curves

def load_images(meanteacher_image_folder):
    valid_images = [".jpg",".gif",".png",".tga",".jpeg"]
    imageList = []
    for filename in sorted(os.listdir(meanteacher_image_folder)):
        ext = os.path.splitext(filename)[1]
        if ext.lower() not in valid_images:
            continue
        imageList.append(meanteacher_image_folder+filename)

    return imageList

def create_training_instances(
    train_annot_folder,
    train_image_folder,
    train_cache,
    valid_annot_folder,
    valid_image_folder,
    valid_cache,
    labels,
):

    # parse annotations of the training set
    train_ints, train_labels = parse_voc_annotation(train_annot_folder, train_image_folder, train_cache, labels)

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(valid_annot_folder):
        valid_ints, valid_labels = parse_voc_annotation(valid_annot_folder, valid_image_folder, valid_cache, labels)
    else:
        print("valid_annot_folder not exists. Spliting the trainining set.")

        train_valid_split = int(0.8*len(train_ints))
        np.random.seed(0)
        np.random.shuffle(train_ints)
        np.random.seed()

        valid_ints = train_ints[train_valid_split:]
        train_ints = train_ints[:train_valid_split]

    # compare the seen labels with the given labels in config.json
    if len(labels) > 0:
        overlap_labels = set(labels).intersection(set(train_labels.keys()))

        print('Seen labels: \t'  + str(train_labels) + '\n')
        print('Given labels: \t' + str(labels))

        # return None, None, None if some given label is not in the dataset
        if len(overlap_labels) < len(labels):
            print('Some labels have no annotations! Please revise the list of labels in the config.json.')
            return None, None, None
    else:
        print('No labels are provided. Train on all seen labels.')
        print(train_labels)
        labels = train_labels.keys()

    max_box_per_image = max([len(inst['object']) for inst in (train_ints + valid_ints)])

    return train_ints, valid_ints, sorted(labels), max_box_per_image

def create_callbacks(saved_weights_name, tensorboard_logs, model_to_save):
    makedirs(tensorboard_logs)
    
    early_stop = EarlyStopping(
        monitor     = 'loss', 
        min_delta   = 0.01, 
        patience    = 20, 
        mode        = 'min', 
        verbose     = 1
    )
    checkpoint = CustomModelCheckpoint(
        model_to_save   = model_to_save,
        filepath        = saved_weights_name,# + '{epoch:02d}.h5', 
        monitor         = 'loss', 
        verbose         = 1, 
        save_best_only  = True, 
        mode            = 'min', 
        period          = 1
    )
    reduce_on_plateau = ReduceLROnPlateau(
        monitor  = 'loss',
        factor   = 0.1,
        patience = 2,
        verbose  = 1,
        mode     = 'min',
        min_delta  = 0.01,
        cooldown = 0,
        min_lr   = 0
    )
    tensorboard = CustomTensorBoard(
        log_dir                = tensorboard_logs,
        write_graph            = True,
        write_images           = True,
    )    
    return [early_stop, checkpoint, reduce_on_plateau, tensorboard]

def create_model(
    nb_class, 
    anchors, 
    max_box_per_image, 
    max_grid, batch_size, 
    warmup_batches, 
    ignore_thresh, 
    multi_gpu, 
    saved_weights_name, 
    pretrained_weights,
    lr,
    grid_scales,
    obj_scale,
    noobj_scale,
    xywh_scale,
    class_scale  
):
    if multi_gpu > 1:
        with tf.device('/cpu:0'):
            template_model, infer_model = create_yolov3_model(
                nb_class            = nb_class, 
                anchors             = anchors, 
                max_box_per_image   = max_box_per_image, 
                max_grid            = max_grid, 
                batch_size          = batch_size//multi_gpu, 
                warmup_batches      = warmup_batches,
                ignore_thresh       = ignore_thresh,
                grid_scales         = grid_scales,
                obj_scale           = obj_scale,
                noobj_scale         = noobj_scale,
                xywh_scale          = xywh_scale,
                class_scale         = class_scale
            )
    else:
        template_model, infer_model = create_yolov3_model(
            nb_class            = nb_class, 
            anchors             = anchors, 
            max_box_per_image   = max_box_per_image, 
            max_grid            = max_grid, 
            batch_size          = batch_size, 
            warmup_batches      = warmup_batches,
            ignore_thresh       = ignore_thresh,
            grid_scales         = grid_scales,
            obj_scale           = obj_scale,
            noobj_scale         = noobj_scale,
            xywh_scale          = xywh_scale,
            class_scale         = class_scale
        )  

    # load the pretrained weight if exists, otherwise load the backend weight only
    if os.path.exists(saved_weights_name): 
        print("\nLoading pretrained weights "+saved_weights_name+".\n")
        template_model.load_weights(saved_weights_name)
    elif os.path.exists(pretrained_weights): 
        print("\nLoading original pretrained weights "+pretrained_weights+".\n")
        template_model.load_weights(pretrained_weights)
    else:
        print("\nFine-tunning backend.h5.\n")
        template_model.load_weights("backend.h5", by_name=True)       

    if multi_gpu > 1:
        train_model = multi_gpu_model(template_model, gpus=multi_gpu)
    else:
        train_model = template_model      

    optimizer = Adam(lr=lr, clipnorm=0.001)
    train_model.compile(loss=dummy_loss, optimizer=optimizer)             

    return train_model, infer_model

def _main_(args):
    config_path = args.conf

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations 
    ###############################
    train_ints, valid_ints, labels, max_box_per_image = create_training_instances(
        config['train']['train_annot_folder'],
        config['train']['train_image_folder'],
        config['train']['cache_name'],
        config['valid']['valid_annot_folder'],
        config['valid']['valid_image_folder'],
        config['valid']['cache_name'],
        config['model']['labels']
    )
    
    ###############################
    #   Parse the annotations 
    ###############################
    train2_ints, valid2_ints, labels2, max_box_per_image2 = create_training_instances(
        config['meanteacher']['train_annot_folder'],
        config['meanteacher']['train_image_folder'],
        config['meanteacher']['train_cache_name'],
        config['meanteacher']['valid_annot_folder'],
        config['meanteacher']['valid_image_folder'],
        config['meanteacher']['valid_cache_name'],
        config['model']['labels']
    )

    ###############################
    #   Create the model 
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    multi_gpu = len(config['train']['gpus'].split(','))

    train_model, infer_model = create_model(
        nb_class            = len(labels), 
        anchors             = config['model']['anchors'], 
        max_box_per_image   = max_box_per_image, 
        max_grid            = [config['model']['max_input_size'], config['model']['max_input_size']], 
        batch_size          = config['train']['batch_size'], 
        warmup_batches      = 0,
        ignore_thresh       = config['train']['ignore_thresh'],
        multi_gpu           = multi_gpu,
        saved_weights_name  = config['train']['saved_weights_name'],
        pretrained_weights  = config['train']['pretrained_weights'],
        lr                  = config['train']['learning_rate'],
        grid_scales         = config['train']['grid_scales'],
        obj_scale           = config['train']['obj_scale'],
        noobj_scale         = config['train']['noobj_scale'],
        xywh_scale          = config['train']['xywh_scale'],
        class_scale         = config['train']['class_scale'],
    )


    ###############################
    #   Create the generators 
    ###############################    
    train_generator = BatchGenerator(
        instances           = train_ints, 
        anchors             = config['model']['anchors'],   
        labels              = labels,        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = max_box_per_image,
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],   
        shuffle             = True, 
        jitter              = 0.3, 
        norm                = normalize
    )
    
    valid_generator = BatchGenerator(
        instances           = valid_ints, 
        anchors             = config['model']['anchors'],   
        labels              = labels,        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = max_box_per_image,
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],   
        shuffle             = True, 
        jitter              = 0.0, 
        norm                = normalize
    )

    ###############################
    #  Get images without annotations
    ###############################
    show = True
    percentual_real = 0.75

    pse_train_imgs, pse_train_labels = parse_voc_annotation(config['meanteacher']['train_annot_folder'],
                                                            config['meanteacher']['train_image_folder'],
                                                            config['meanteacher']['train_cache_name'],
                                                            config['model']['labels'])

    pse_valid_imgs, pse_valid_labels = parse_voc_annotation(config['meanteacher']['valid_annot_folder'],
                                                            config['meanteacher']['valid_image_folder'],
                                                            config['meanteacher']['valid_cache_name'],
                                                            config['model']['labels'])

    pse_train_valid_imgs = pse_train_imgs
    pse_train_valid_imgs.extend(pse_valid_imgs)


    pseudo_images = []
    real_images = []
    count = 0
    for image in pse_train_valid_imgs:
        fileName = image['filename']

        raw_image = cv2.imread(fileName)
        obj_thresh=0.5
        nms_thresh=0.45
        net_h=416
        net_w=416
        score_threshold = 0.80
        
        prediction  = get_yolo_boxes(infer_model,[raw_image], net_h, net_w, train_generator.get_anchors(), obj_thresh, nms_thresh)
        if len(prediction[0]) > 0:

            raw_height, raw_width, _ = raw_image.shape
            img = PredictToBndBox(fileName, raw_height, raw_width, prediction[0], labels, score_threshold)

            if len(img['object'])>0:
                pseudo_images += [img]

                count += 1
                if show:
                    print(count, fileName)
                    print(img['object'])
                    print(image['object'])
                else:
                    print(count, end ="", flush=True)
                    print('\r', end='')

    
    

    #Add source images 
    count_pseudo_images = len(pseudo_images)
    np.random.shuffle(train_ints)
    rnd_source_images = int(count_pseudo_images * percentual_real)
    pseudo_images.extend(train_ints[:rnd_source_images])
    np.random.shuffle(pseudo_images)

    print("Pseudo Images ",count_pseudo_images)
    print("Real Images Added",rnd_source_images)
    print("Total Images to train",len(pseudo_images))


    ###############################    
    train_generator = BatchGenerator(
        instances           = pseudo_images, 
        anchors             = config['model']['anchors'],   
        labels              = labels,        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = max_box_per_image,
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],   
        shuffle             = True, 
        jitter              = 0.3, 
        norm                = normalize
    )

    ###############################
    #   Kick off the training
    ###############################
    callbacks = create_callbacks(config['train']['saved_weights_name'], config['train']['tensorboard_dir'], infer_model)

    train_model.fit_generator(
        generator        = train_generator, 
        steps_per_epoch  = len(train_generator) * config['train']['train_times'], 
        epochs           = config['train']['nb_epochs'] + config['train']['warmup_epochs'], 
        verbose          = 2 if config['train']['debug'] else 1,
        callbacks        = callbacks, 
        workers          = 4,
        max_queue_size   = 8
    )

    # make a GPU version of infer_model for evaluation
    if multi_gpu > 1:
        infer_model = load_model(config['train']['saved_weights_name'])

    ###############################
    #   Run the evaluation
    ###############################   
    # compute mAP for all the classes
    average_precisions = evaluate(infer_model, valid_generator)

    # print the score
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))           

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='train and evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')   

    args = argparser.parse_args()
    _main_(args)
