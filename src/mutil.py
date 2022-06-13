import matplotlib.pyplot as plt

## 
def return_shape(data):
    return data[0][0].shape[1:]

def dict_result(train_loss, train_acc, test_loss, test_acc):
    results = {
        "train_loss" : train_loss,
        "train_acc" : train_acc,
        "test_loss" : test_loss,
        "test_acc" : test_acc
    }
    return results

def get_input_shape(target_size):
    return (target_size[0], target_size[1], 3)

def plot_history(history, title="loss", history_type="loss", path_figure=""):
    val = "val_" + history_type

    if len(title.split(".")) == 1:
        title += ".jpg"
    save_path = path_figure + title

    if type(history) == dict:
        h = history
    else:
        h = history.history

    plt.plot(h[history_type])
    plt.plot(h[val])
    plt.title(history_type)
    plt.ylabel(history_type)
    plt.xlabel("Epochs")
    plt.legend(["Training", "Validation"], loc=0)
    plt.savefig(save_path)
    plt.clf()

def save_txt(result = {}, title="result", path_txt="txtfiles/"):
    key_list = ["train_loss", "train_acc", "test_loss", "test_acc"]

    if len(title.split(".")) == 1:
        title += ".txt"
    save_path = path_txt + title

    with open(save_path, "w") as f:
        for key in key_list:
            string = ""
            string += (key + ": ") 
            string += str(result[key])
            string += "\n"
            f.write(string)

def str_to_tuple(i):
    shape = []
    for s in (i.split(",")):
        shape.append(int(s))

    return tuple(shape)

def highlight_string(s):
    print("="*30)
    print("="*30)
    print("="*30)
    print(s)
    print("="*30)
    print("="*30)
    print("="*30)
