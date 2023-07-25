import numpy as np
import os
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from sklearn.metrics import confusion_matrix, f1_score,  roc_curve, auc
from sklearn.cluster import MiniBatchKMeans, KMeans
from imblearn.over_sampling import KMeansSMOTE
from sklearn.model_selection import StratifiedKFold

# normalize
scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def cnn1d_LSTM(inputs_shapes):
    inputs = tf.keras.Input(shape=inputs_shapes)
    x = tf.keras.layers.Conv1D(filters=8, kernel_size=10, strides=1, padding='valid', activation=tf.nn.relu)(inputs)
    f1 = tf.keras.layers.MaxPool1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(filters=8, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu)(f1)
    x = tf.keras.layers.Conv1D(filters=8, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu)(x)
    added1 = tf.keras.layers.add([f1, x])
    x = tf.keras.activations.relu(added1)

    x = tf.keras.layers.Conv1D(filters=16, kernel_size=10, strides=1, padding='valid', activation=tf.nn.relu)(x)
    f2 = tf.keras.layers.MaxPool1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)(f2)
    x = tf.keras.layers.Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)(x)
    added2 = tf.keras.layers.add([f2, x])
    x = tf.keras.activations.relu(added2)

    f3 = tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=1, padding='valid', activation=tf.nn.relu)(x)

    x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)(f3)
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)(x)
    added3 = tf.keras.layers.add([f3, x])
    x = tf.keras.activations.relu(added3)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(units=2, activation=tf.nn.softmax)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def lr_schedule(epoch, lr):
    if epoch > 70 and (epoch - 1) % 10 == 0:
        lr *= 0.1
    print("Learning rate: ", lr)
    return lr


if __name__ == "__main__":
    with open(os.path.join("apnea_spo2_segment.pkl"),
              'rb') as f:  # read preprocessing result
        apnea_spo2 = pickle.load(f)
    o_UCD = apnea_spo2["o_UCD"]
    y_UCD = apnea_spo2["y_UCD"]
    o_UCD = scaler(o_UCD)
    o_UCD = np.array(o_UCD)
    y_UCD = np.array(y_UCD)

    cv = StratifiedKFold(n_splits=10)
    classifier = cnn1d_LSTM(inputs_shapes=(480, 1))

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(10, 10))
    i = 0
    for train_index, test_index in cv.split(o_UCD, y_UCD):
        X_train, X_test = o_UCD[train_index], o_UCD[test_index]
        y_train, y_testa = y_UCD[train_index], y_UCD[test_index]

        ros = KMeansSMOTE(
            kmeans_estimator=MiniBatchKMeans(n_init=1, random_state=0), random_state=42
        )
        X_res, y_resa = ros.fit_resample(X_train, y_train)
        y_res = tf.keras.utils.to_categorical(y_resa, num_classes=2)
        y_test = tf.keras.utils.to_categorical(y_testa, num_classes=2)

        classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        checkpoint = ModelCheckpoint('weights.best.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        stopping = EarlyStopping(monitor='val_acc', patience=5)
        lr_scheduler = LearningRateScheduler(lr_schedule)
        callbacks_list = [lr_scheduler, checkpoint, stopping]
        classifier.fit(X_res, y_res, batch_size=128, epochs=100, callbacks=callbacks_list, validation_split=0.2,
                       verbose=1)
        probas_ = classifier.predict(X_test)

        y_true, y_pred = np.argmax(y_test, axis=-1), np.argmax(classifier.predict(X_test, batch_size=1024, verbose=1),
                                                               axis=-1)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        print(type(y_true))
        print(type(y_pred))
        print(y_true.shape)
        print(y_pred.shape)
        C = confusion_matrix(y_true, y_pred, labels=(1, 0))
        TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]
        acc, sn, sp, pre = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP), TP / (
                    TP + FP)
        f1 = f1_score(y_true, y_pred, average='binary')
        print("acc: {}, sn: {}, sp: {}, pre: {} ,f1: {}".format(acc, sn, sp, pre, f1))

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_testa, probas_[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i + 1, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #     plt.title('Cross-Validation ROC of SVM', fontsize=18)
    plt.legend(loc="lower right", prop={'size': 15})
    plt.savefig('five.png', transparent=True, dpi=300)
    plt.show()
