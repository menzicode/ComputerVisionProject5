# Import statements
import sys
import torch
import torchvision as tv 
import matplotlib.pyplot as plt
import TrainModel as tm
import cv2


# Function to print output evaluation and plot images with predictions 
# Parameters: image data, image target labeles, number of images evaluated
def evaluate(img_data, img_targets, n_examples):
    # Load network
    network = tm.MyNetwork()
    network.load_state_dict(torch.load('results/model.pth'))
    network.eval()

    # Generate output
    output = network(img_data)
    predictions = []

    # Evaluate output and print results 
    for i in range(n_examples):
        op = output[i]
        print("Example {}".format(i))
        max_index = 0
        max_output = max(op)
        for j in range(10):
            if (op[j] == max_output):
                max_index = j
                predictions.append(j)
            print("Output value for number {num}: {oput:.2f}".format(num=j,oput=op[j]))
        print("Max output value {m:.2f} found at index {ind}".format(m=max(op), ind=max_index))
        print("Actual example label: {}".format(img_targets[i]))
        if (img_targets[i] == max_index):
            print("Accurate: Yes\n")
        if (img_targets[i] != max_index):
            print("Accurate: No\n")
    
    # Plot images and with predictions 
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.tight_layout()
        plt.imshow(img_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(predictions[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

# Loads hand drawn images, processes them to be readable by network, and returns data and targets
def prep_hand_draw():
    imgnames = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    for i in range(10):
        path = 'hand_drawn/' + imgnames[i] + '.jpg'
        img = cv2.imread(path)
        thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[1]
        newimg = cv2.split(thresh)[0]
        newpath = 'hand_drawn_data/' + str(i) + '/' + str(i) + '.png'
        cv2.imwrite(newpath, newimg)
    dataset = tv.datasets.ImageFolder('hand_drawn_data/', transform=tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Grayscale(), tv.transforms.Normalize((0.1307,), (0.3081,))]))
    set_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)
    img_data, img_targets = next(iter(set_loader))
    return img_data, img_targets

def main(argv):
    handrawn = True

    # number of images to evaluate
    n_examples = 10

    #Load the testing data
    if (handrawn):
        img_data, img_targets = prep_hand_draw() 
        evaluate(img_data, img_targets, n_examples)

    else:
        testLoader = tm.test_loader(n_examples)
        loader_set = enumerate(testLoader)
        img_data, img_targets = next(loader_set)[1]
        evaluate(img_data, img_targets, n_examples)

    return

if __name__ == "__main__":
    main(sys.argv)