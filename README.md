# TimeSeriesAnomalyUsingTransformations
Detect anomalies in time series by classifying transformation on the input


### Train
```
# Train on 1d  power-sinus data with permutations as learned transformation 
python3 main.py --data_type 1d --train_dir train_1d --transformations_type Permutations --func_type Power-Sinus

# Train on 2d data with affine transformations and varying shapes as anomaly
 python3 main.py --data_type 2d --train_dir train_2d --transformations_type Affine --anomaly_type Shapes

```
### Test
```
just add --test to the exact same command as the above to perform a test
```
### Create Debug images

```
just add --debug to the exact same command as the above to generate debug images
```

### Citation
This repository is a student project inspired by the paper
Deep Anomaly Detection Using Geometric Transformations


```
@article{golan2018deep,
  title={Deep Anomaly Detection Using Geometric Transformations},
  author={Golan, Izhak and El-Yaniv, Ran},
  journal={arXiv preprint arXiv:1805.10917},
  year={2018}
}
```
https://github.com/izikgo/AnomalyDetectionTransformations
