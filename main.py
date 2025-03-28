import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier as ORC
from sklearn.model_selection import train_test_split
import math
import warnings

# Silence warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn


class Evaluator:
    """
    A class for evaluating the system's performance.
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
        
        # Number of bins for the histograms
        bins = np.linspace(0, 1, 20)
        
        # Plotting the histogram for genuine scores
        plt.hist(
            self.genuine_scores,
            bins=bins,
            color='green',
            alpha=0.6,
            lw=2,
            histtype='step',
            hatch='\\',
            edgecolor='green',
            fill=True,
            label='Genuine'
        )
        
        # Plotting the histogram for impostor scores
        plt.hist(
            self.impostor_scores,
            bins=bins,
            color='red',
            alpha=0.6,
            lw=2,
            histtype='step',
            hatch='/',
            edgecolor='red',
            fill=True,
            label='Impostor'
        )
        
        # Set the x-axis limit
        plt.xlim([-0.05, 1.05])
        
        # Add grid lines
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        
        # Add legend
        plt.legend(
            loc='upper left',
            fontsize=12
        )
        
        # Set x and y-axis labels
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

        # Scale the y-axis logarithmically for better visualization of the score distribution
        plt.yscale('log')
        
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
        
        # Calculating EER // average of FPR and FNR at this index
        EER = (FPR[min_index] + FNR[min_index]) / 2.0
        
        return EER

    def plot_det_curve(self, FPR, FNR):
        """
        Plot the Detection Error Tradeoff (DET) curve.
        Parameters:
         - FPR (list or array-like): False Positive Rate values.
         - FNR (list or array-like): False Negative Rate values.
        """
        
        # Calculate EER using the get_EER method
        EER = self.get_EER(FPR, FNR)
        
        # Create a new figure
        plt.figure()
        
        # Plot the DET Curve
        plt.plot(
            FPR,
            FNR,
            lw=2,
            color='blue'
        )
        
        # Text displaying the EER value on the graph
        plt.text(EER + 0.07, EER + 0.07, f"EER = {EER:.5f}", style='italic', fontsize=12,
                 bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
        plt.plot([0, 1], [0, 1], '--', lw=0.5, color='black')
        plt.scatter([EER], [EER], c="black", s=100)
        
        # Set the x and y-axis limits
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
        
        # Set x and y-axis labels
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
        
        # Title to the plot with EER value and system title
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
        
        # Save the plot as an image
        plt.savefig('det_curve_plot_(%s).png' % self.plot_title, dpi=300, bbox_inches="tight")
        
        # Display
        plt.show()
        
        # Close
        plt.close()
    
        return

    def plot_roc_curve(self, FPR, TPR):
        """
        Plot the Receiver Operating Characteristic (ROC) curve.
        Parameters:
        - FPR (list or array-like): False Positive Rate values.
        - TPR (list or array-like): True Positive Rate values.
        """
        
        # New figure
        plt.figure()
        
        # Plot the ROC curve
        plt.plot(FPR, TPR, lw=2, color='blue')
        
        # Calculate the Area Under the Curve (AUC)
        AUC = metrics.auc(FPR, TPR)
        
        # Set x and y-axis limits
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        
        # Add grid lines for better readability
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        
        # Remove the top and right spines for a cleaner appearance
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Set x and y-axis labels
        plt.xlabel('False Pos. Rate', fontsize=12, weight='bold')
        plt.ylabel('True Pos. Rate', fontsize=12, weight='bold')
        
        # Add a title to the plot with AUC value and system title
        plt.title('Receiver Operating Characteristic Curve \nAUC = %.5f\nSystem %s' % 
                 (AUC, self.plot_title), fontsize=15, weight='bold')
        
        # Set font size for x and y-axis
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
        # Save the plot as Image
        plt.savefig('roc_curve_plot_(%s).png' % self.plot_title, dpi=300, bbox_inches="tight")
        
        # Display
        plt.show()
        
        # Close the figure
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
        FPR = []
        FNR = []
        TPR = []
        
        # Iterate through thresholds
        for threshold in self.thresholds:
            # Calculate true positives, false positives, true negatives, false negatives
            TP = sum(1 for score in self.genuine_scores if score >= threshold)
            FP = sum(1 for score in self.impostor_scores if score >= threshold)
            TN = sum(1 for score in self.impostor_scores if score < threshold)
            FN = sum(1 for score in self.genuine_scores if score < threshold)
            
            # Compute rates and adding epsilon to prevent division by zero
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
    Extract features from the facial landmark data with improved accuracy.
    """
    num_samples = X.shape[0]
    num_landmarks = X.shape[1]
    
    # Initialize feature matrix
    all_features = []
    
    # For each sample, compute features between pairs of landmarks
    for k in range(num_samples):
        features_k = []
        
        # Calculate pairwise features for landmarks
        for i in range(num_landmarks):
            for j in range(i+1, num_landmarks):
                p1 = X[k, i]
                p2 = X[k, j]
                
                # Euclidean distance (normalized)
                features_k.append(math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))
                # Absolute difference in x-coordinate
                features_k.append(abs(p1[0] - p2[0]))
                # Absolute difference in y-coordinate
                features_k.append(abs(p1[1] - p2[1]))
                # Angle between points (new feature)
                angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                features_k.append(angle)
        
        all_features.append(features_k)
    
    # Convert to numpy array
    features = np.array(all_features)
    
    # Standardize features (z-score normalized)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1e-10  # Avoiding division by zero
    normalized_features = (features - mean) / std
    
    return normalized_features


def biometric_system():
    """
    Main function for the biometric authentication system using k-NN and ORC on Caltech dataset.
    """
    # Set random seed for reproducibility with score selection
    np.random.seed(42)
    print("Loading data...")
    try:
        X = np.load("X-68-Caltech.npy")  # Loading Caltech Files
        y = np.load("y-68-Caltech.npy")
        print(f"Loaded dataset with {X.shape[0]} samples, {X.shape[1]} landmarks")
    except FileNotFoundError: # Error Handling
        print("Error: Dataset files not found")
        return
    
    # Step 2: Split data into training and testing
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples, Testing set: {X_test.shape[0]} samples")
    
    # Step 3: Extract features from training data
    print("Extracting features from training data...")
    X_train_features = extract_features(X_train)
    print(f"Extracted {X_train_features.shape[1]} features per sample")
    
    # Step 4: Create and train with k-NN
    print("Training k-NN classifier with OneVsRest strategy...")
    # Use a higher k value and distance weighting to get smoother probability outputs
    clf = ORC(KNeighborsClassifier(n_neighbors=9, weights='distance', metric='minkowski', p=2, algorithm='auto'))
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

        true_class_idx = np.where(clf.classes_ == test_label)[0][0]
        
        # Get the probability score for the true class (genuine score)
        genuine_score = matching_scores[i, true_class_idx]
        genuine_scores.append(genuine_score)
        
        # Get all impostor scores for this test sample
        for j, prob in enumerate(matching_scores[i]):
            if j != true_class_idx:
                impostor_scores.append(prob)
    
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    
    print(f"Collected {len(genuine_scores)} genuine scores and {len(impostor_scores)} impostor scores")
    
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
    
    return clf, X_test_features, y_test


if __name__ == "__main__":
    
    # Run
    
    biometric_system()
