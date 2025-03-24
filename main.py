import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier as ORC
from sklearn.model_selection import train_test_split
import math
import warnings

# Suppress warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn


class Evaluator:
    """
    A class for evaluating a biometric system's performance.
    """

    def __init__(self, 
                 num_thresholds, 
                 genuine_scores, 
                 impostor_scores, 
                 plot_title, 
                 epsilon=1e-12):
        """
        Initialize the Evaluator object.

        Parameters:
        - num_thresholds (int): Number of thresholds to evaluate.
        - genuine_scores (array-like): Genuine scores for evaluation.
        - impostor_scores (array-like): Impostor scores for evaluation.
        - plot_title (str): Title for the evaluation plots.
        - epsilon (float): A small value to prevent division by zero.
        """
        self.num_thresholds = num_thresholds
        self.thresholds = np.linspace(-0.1, 1.1, num_thresholds)
        self.genuine_scores = genuine_scores
        self.impostor_scores = impostor_scores
        self.plot_title = plot_title
        self.epsilon = epsilon

    def get_dprime(self):
        """
        Calculate the d' (d-prime) metric.

        Returns:
        - float: The calculated d' value.
        """
        genuine_mean = np.mean(self.genuine_scores)
        impostor_mean = np.mean(self.impostor_scores)
        genuine_std = np.std(self.genuine_scores)
        impostor_std = np.std(self.impostor_scores)
        
        # Calculate numerator and denominator for d-prime
        x = np.abs(genuine_mean - impostor_mean)
        y = np.sqrt(0.5 * (genuine_std**2 + impostor_std**2))
        
        return x / (y + self.epsilon)

    def plot_score_distribution(self):
        """
        Plot the distribution of genuine and impostor scores.
        """
        plt.figure(figsize=(10, 8))
        
        # Define number of bins for the histograms
        bins = np.linspace(0, 1, 20)
        
        # Plot the histogram for genuine scores
        plt.hist(
            self.genuine_scores,
            bins=bins,
            color='green',
            lw=2,
            histtype='step',
            hatch='\\',
            label='Genuine'
        )
        
        # Plot the histogram for impostor scores
        plt.hist(
            self.impostor_scores,
            bins=bins,
            color='red',
            lw=2,
            histtype='step',
            hatch='/',
            label='Impostor'
        )
        
        # Set the x-axis limit to ensure the histogram fits within the correct range
        plt.xlim([-0.05, 1.05])
        
        # Add grid lines for better readability
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        
        # Add legend to the upper left corner with a specified font size
        plt.legend(
            loc='upper left',
            fontsize=12
        )
        
        # Set x and y-axis labels with specified font size and weight
        plt.xlabel(
            'Similarity Score',
            fontsize=12,
            weight='bold'
        )
        
        plt.ylabel(
            'Frequency',
            fontsize=12,
            weight='bold'
        )
        
        # Remove the top and right spines for a cleaner appearance
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Set font size for x and y-axis ticks
        plt.xticks(
            fontsize=10
        )
        
        plt.yticks(
            fontsize=10
        )
        
        # Add a title to the plot with d-prime value and system title
        plt.title('Score Distribution Plot\nd-prime= %.2f\nSystem %s' % 
                  (self.get_dprime(), 
                   self.plot_title),
                  fontsize=15,
                  weight='bold')
        
       
        # Save the figure before displaying it
        plt.savefig('score_distribution_plot_(%s).png' % self.plot_title, dpi=300, bbox_inches="tight")
        
        # Display the plot after saving
        plt.show()
        
        # Close the figure to free up resources
        plt.close()

        return

    def get_EER(self, FPR, FNR):
        """
        Calculate the Equal Error Rate (EER).
    
        Parameters:
        - FPR (list or array-like): False Positive Rate values.
        - FNR (list or array-like): False Negative Rate values.
    
        Returns:
        - float: Equal Error Rate (EER).
        """
        # Convert to numpy arrays if they're not already
        FPR = np.array(FPR)
        FNR = np.array(FNR)
        
        # Find the index where FPR and FNR are closest to each other
        abs_diff = np.abs(FPR - FNR)
        min_index = np.argmin(abs_diff)
        
        # Calculate EER as the average of FPR and FNR at this index
        EER = (FPR[min_index] + FNR[min_index]) / 2.0
        
        return EER

    def plot_det_curve(self, FPR, FNR):
        """
        Plot the Detection Error Tradeoff (DET) curve.
        Parameters:
         - FPR (list or array-like): False Positive Rate values.
         - FNR (list or array-like): False Negative Rate values.
        """
        
        # Calculate the Equal Error Rate (EER) using the get_EER method
        EER = self.get_EER(FPR, FNR)
        
        # Create a new figure for plotting
        plt.figure()
        
        # Plot the Detection Error Tradeoff Curve
        plt.plot(
            FPR,
            FNR,
            lw=2,
            color='blue'
        )
        
        # Add a text annotation for the EER point on the curve
        plt.text(EER + 0.07, EER + 0.07, "EER", style='italic', fontsize=12,
                 bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
        plt.plot([0, 1], [0, 1], '--', lw=0.5, color='black')
        plt.scatter([EER], [EER], c="black", s=100)
        
        # Set the x and y-axis limits to ensure the plot fits within the range 
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        
        # Add grid lines for better readability
        plt.grid(
            color='gray',
            linestyle='--',
            linewidth=0.5
        )
        
        # Remove the top and right spines for a cleaner appearance
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Set x and y-axis labels with specified font size and weight
        plt.xlabel(
            'False Pos. Rate',
            fontsize=12,
            weight='bold'
        )
        
        plt.ylabel(
            'False Neg. Rate',
            fontsize=12,
            weight='bold'
        )
        
        # Add a title to the plot with EER value and system title
        plt.title(
            'Detection Error Tradeoff Curve \nEER = %.5f\nSystem %s' % (EER, self.plot_title),
            fontsize=15,
            weight='bold'
        )
        
        # Set font size for x and y-axis ticks
        plt.xticks(
            fontsize=10
        )
        
        plt.yticks(
            fontsize=10
        )
        
        # Save the plot as an image file
        plt.savefig('det_curve_plot_(%s).png' % self.plot_title, dpi=300, bbox_inches="tight")
        
        # Display the plot
        plt.show()
        
        # Close the plot to free up resources
        plt.close()
    
        return

    def plot_roc_curve(self, FPR, TPR):
        """
        Plot the Receiver Operating Characteristic (ROC) curve.
        Parameters:
        - FPR (list or array-like): False Positive Rate values.
        - TPR (list or array-like): True Positive Rate values.
        """
        
        # Create a new figure for the ROC curve
        plt.figure()
        
        # Plot the ROC curve using FPR and TPR
        plt.plot(FPR, TPR, lw=2, color='blue')
        
        # Calculate the Area Under the Curve (AUC)
        AUC = metrics.auc(FPR, TPR)
        
        # Set x and y-axis limits to ensure the plot fits within the range
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        
        # Add grid lines for better readability
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        
        # Remove the top and right spines for a cleaner appearance
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Set x and y-axis labels with specified font size and weight
        plt.xlabel('False Pos. Rate', fontsize=12, weight='bold')
        plt.ylabel('True Pos. Rate', fontsize=12, weight='bold')
        
        # Add a title to the plot with AUC value and system title
        plt.title('Receiver Operating Characteristic Curve \nAUC = %.5f\nSystem %s' % 
                 (AUC, self.plot_title), fontsize=15, weight='bold')
        
        # Set font size for x and y-axis ticks
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
        # Save the plot as a PNG file
        plt.savefig('roc_curve_plot_(%s).png' % self.plot_title, dpi=300, bbox_inches="tight")
        
        # Display the plot
        plt.show()
        
        # Close the figure to free up resources
        plt.close()
 
        return

    def compute_rates(self):
        """
        Compute false positive, false negative, and true positive rates across thresholds.
        
        Returns:
        - tuple: (FPR, FNR, TPR) - Lists of false positive rates, false negative rates,
                and true positive rates for each threshold.
        """
        # Initialize lists for rates
        FPR = []  # False Positive Rate
        FNR = []  # False Negative Rate 
        TPR = []  # True Positive Rate
        
        # Iterate through thresholds
        for threshold in self.thresholds:
            # Calculate true positives, false positives, true negatives, false negatives
            TP = sum(1 for score in self.genuine_scores if score >= threshold)
            FP = sum(1 for score in self.impostor_scores if score >= threshold)
            TN = sum(1 for score in self.impostor_scores if score < threshold)
            FN = sum(1 for score in self.genuine_scores if score < threshold)
            
            # Compute rates (add epsilon to prevent division by zero)
            fpr = FP / (FP + TN + self.epsilon)
            fnr = FN / (FN + TP + self.epsilon)
            tpr = TP / (TP + FN + self.epsilon)
            
            # Append to lists
            FPR.append(fpr)
            FNR.append(fnr)
            TPR.append(tpr)
        
        return FPR, FNR, TPR


def extract_features(X):
    """
    Extract features from the facial landmark data.
    
    Parameters:
    - X (numpy.ndarray): Input data containing facial landmarks.
    
    Returns:
    - numpy.ndarray: Feature matrix containing pairwise Euclidean distances and absolute differences.
    """
    num_samples = X.shape[0]
    num_landmarks = X.shape[1]
    
    # Initialize list to store features for each sample
    all_features = []
    
    # For each sample, compute features between pairs of landmarks
    for k in range(num_samples):
        features_k = []
        
        # Calculate pairwise features for landmarks
        for i in range(num_landmarks):
            for j in range(i+1, num_landmarks):
                p1 = X[k, i]
                p2 = X[k, j]
                
                # Euclidean distance
                features_k.append(math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))
                # Absolute difference in x-coordinate
                features_k.append(abs(p1[0] - p2[0]))
                # Absolute difference in y-coordinate
                features_k.append(abs(p1[1] - p2[1]))
        
        all_features.append(features_k)
    
    return np.array(all_features)


def main():
    """
    Main function to run system B simulation.
    """
    # Set the random seed to 1
    np.random.seed(1)
    
    # Use system B
    system = "B"
    
    # Generate genuine scores with mean around 0.6 and std of 0.08
    genuine_mean = 0.6
    genuine_std = 0.08
    genuine_scores = np.random.normal(genuine_mean, genuine_std, 400)
    genuine_scores = np.clip(genuine_scores, 0, 1)
    
    # Generate impostor scores with mean around 0.25 and std of 0.15
    impostor_mean = 0.25
    impostor_std = 0.15
    impostor_scores = np.random.normal(impostor_mean, impostor_std, 1600)
    impostor_scores = np.clip(impostor_scores, 0, 1)
    
    # Creating an instance of the Evaluator class
    evaluator = Evaluator(
        epsilon=1e-12,
        num_thresholds=200,
        genuine_scores=genuine_scores,
        impostor_scores=impostor_scores,
        plot_title="%s" % system
    )
    
    # Generate the FPR, FNR, and TPR using 200 threshold values
    FPR, FNR, TPR = evaluator.compute_rates()

    # Plot the score distribution
    evaluator.plot_score_distribution()
            
    # Plot the DET curve and include the EER in the plot's title
    evaluator.plot_det_curve(FPR, FNR)
    
    # Plot the ROC curve
    evaluator.plot_roc_curve(FPR, TPR)


def biometric_system():
    """
    Main function for the biometric authentication system using k-NN and ORC on Caltech dataset.
    """
    # Step 1: Load the Caltech dataset with 5 facial landmarks
    print("Loading data...")
    try:
        X = np.load("X-68-Caltech.npy")  # Use your dataset file name
        y = np.load("y-68-Caltech.npy")  # Use your dataset file name
    except FileNotFoundError:
        print("Error: Dataset files not found. Please ensure the .npy files are in the current directory.")
        return
    
    # Step 2: Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    
    # Step 3: Extract features from training data
    print("Extracting features from training data...")
    X_train_features = extract_features(X_train)
    
    # Step 4: Create and train a OneVsRest classifier with k-NN
    print("Training k-NN classifier with OneVsRest strategy...")
    clf = ORC(KNeighborsClassifier(n_neighbors=5))
    clf.fit(X_train_features, y_train)
    
    # Step 5: Extract features from test data
    print("Extracting features from test data...")
    X_test_features = extract_features(X_test)
    
    # Step 6: Predict probabilities using the trained classifier
    print("Computing matching scores...")
    matching_scores = clf.predict_proba(X_test_features)
    
    # Step 7: Separate scores into genuine and impostor
    print("Separating genuine and impostor scores...")
    genuine_scores = []
    impostor_scores = []
    
    for i, test_label in enumerate(y_test):
        # Find the index of the true class for this test sample
        true_class_idx = np.where(clf.classes_ == test_label)[0][0]
        
        # Get the probability score for the true class (genuine score)
        genuine_score = matching_scores[i, true_class_idx]
        genuine_scores.append(genuine_score)
        
        # Get probability scores for all other classes (impostor scores)
        for j, prob in enumerate(matching_scores[i]):
            if j != true_class_idx:
                impostor_scores.append(prob)
    
    # Convert to numpy arrays
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    
    # Step 8: Evaluate performance using the Evaluator class
    print("Evaluating system performance...")
    evaluator = Evaluator(
        num_thresholds=200,
        genuine_scores=genuine_scores,
        impostor_scores=impostor_scores,
        plot_title="Caltech Facial Biometrics"
    )
    
    # Generate rates
    FPR, FNR, TPR = evaluator.compute_rates()
    
    # Create evaluation plots
    print("Generating performance plots...")
    evaluator.plot_score_distribution()
    evaluator.plot_det_curve(FPR, FNR)
    evaluator.plot_roc_curve(FPR, TPR)
    
    # Calculate and display system metrics
    print("\nBiometric System Evaluation Results:")
    print("d-prime = %.4f" % evaluator.get_dprime())
    print("EER = %.4f" % evaluator.get_EER(FPR, FNR))
    print("AUC = %.4f" % metrics.auc(FPR, TPR))
    
    # Calculate accuracy
    y_pred = clf.predict(X_test_features)
    accuracy = np.mean(y_pred == y_test)
    print("Accuracy = %.4f" % accuracy)


if __name__ == "__main__":
    # Run the main function to simulate system B
    main()
    
    # If you want to run the biometric system with Caltech dataset,
    # uncomment the following line:
    # biometric_system()
