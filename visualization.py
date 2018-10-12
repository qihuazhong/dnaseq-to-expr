

import matplotlib.pyplot as plt

def plot_train_histories(histories):
    plt.clf()
    train_mae = []
    val_mae = []
    
    loss = []
    val_loss= []
    for history in histories:
        for error in history.history['mean_absolute_error']:
            train_mae.append(error)
        for error in history.history['val_mean_absolute_error']:
            val_mae.append(error)
        for error in history.history['loss']:
            loss.append(error)
        for error in history.history['val_loss']:
            val_loss.append(error)
    epochs = range(1, len(train_mae) + 1)

    plt.plot(epochs, train_mae, 'b', label='Training mae')
    plt.plot(epochs, val_mae, 'r', label='val mae')
    plt.title('mean absolute error')
    plt.legend()
    plt.show()
    
    
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='val loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()



def line_pred_vs_expr(pred_log, expr_log):
    plt.figure(figsize=(14, 8))
    plt.plot(pred_log[:50], label='pred')
    plt.plot(expr_log[:50], label='actual exp')
    plt.legend(['pred', 'exp'], loc='upper left')
    plt.show()


def scatter_pred_vs_expr(pred_log, expr_log):
    plt.figure(figsize=(12, 12))
    plt.xlabel("log prediction")
    plt.ylabel("log actual expression")
    plt.scatter(pred_log, expr_log, alpha=0.3)
    plt.show()

