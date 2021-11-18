from keras import backend as K

# Refer to: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f, axis=-1)
    union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def dice_loss(y_true, y_pred):
    return 1 - (dice_coef(y_true, y_pred))
