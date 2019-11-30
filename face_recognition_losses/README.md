|Loss|Formula|Embedding Viz|Test Accuracy(30 epoch)|
|----|----|----|----------|
|SoftmaxLoss|![softmax_loss](./images/softmax_loss.png)|![softmaxloss](./images/gifs/softmax_loss.gif)|97.71|
|[CenterLoss-ECCV16](https://ydwen.github.io/papers/WenECCV16.pdf)|SoftmaxLoss + ![center_loss](./images/center_loss.png)|![center](./images/gifs/center_loss.gif)|97.22|
|[SphereFace-CVPR17](https://arxiv.org/abs/1704.08063)|![sphere_face_loss1](./images/sphere_face_loss1.png)![sphere_face_loss2](./images/sphere_face_loss2.png)|![sphere](./images/gifs/sphere_face_loss.gif)|98.44|
|[CosFace-CVPR18](https://arxiv.org/abs/1801.09414)|![cosface](./images/cos_face_loss.png)|![cosface](./images/gifs/cos_face_loss.gif)|98.35|
|[ArcFace-CVPR19](https://arxiv.org/abs/1801.07698)|![arcface](./images/arc_face_loss.png)|![arcface](./images/gifs/arc_face_loss.gif)|98.69|

# Train
```
python3 main.py --loss softmax_loss/sphere_face_loss/center_loss/cos_face_loss/arc_face_loss
```
