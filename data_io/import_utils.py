"""
Utility functions for importing data of different formats. E.g., well plat to tidy data, tape station electropherograms, read info from file name.
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# ===========================================
# Extract experiment info from filename
# ===========================================

def get_filename_info(file_name: str, delimiter: str = '_', pattern_mapping: dict = None):
    """
    Extract information about experimental conditions from a movie filename.
    
    The function assumes the filename follows a pattern where various experimental parameters 
    are encoded, using a specified delimiter (default is underscore `_`). 
    
    The function allows for custom extraction of parameters using a pattern mapping where keys
    represent parameter names and values represent their index positions in the filename.
    
    Parameters:
    file_name (str): The name of the movie file containing experimental details.
    delimiter (str): The character used to separate different parts of the filename. Default is `_`.
    pattern_mapping (dict): A dictionary mapping each parameter to its index in the filename parts. 
                            Example: {'date': 0, 'construct': 2, 'treatment': 3, ...}

    Returns:
    dict: A dictionary containing the extracted parameters based on the pattern mapping.
          Keys are the parameter names, values are the extracted values.

    Example:
    >>> get_filename_info("20230115_sample_construct_treatment_50uM_drug_01_rep1_extra_param.tif", 
                          delimiter='_',
                          pattern_mapping={'date': 0, 'construct': 2, 'treatment': 3, 
                                           'concentration': 4, 'compound': 5, 'replicate': 7})
    Output: {'date': '20230115', 'construct': 'construct', 'treatment': 'treatment', 
             'compound': 'drug', 'concentration': 50, 'replicate': 1}
    """

    # Split the filename by the specified delimiter
    parts = file_name.split(delimiter)
    
    # Default pattern mapping if none is provided
    if pattern_mapping is None:
        pattern_mapping = {
            'date': 0, 
            'construct': 2, 
            'treatment': 3, 
            'concentration': 4, 
            'compound': 5, 
            'replicate': 7
        }
    
    filename_info = {}
    
    # Iterate over the pattern mapping and extract the values
    for param, index in pattern_mapping.items():
        try:
            value = parts[index]
            
            # Special handling for concentration and replicate (which are typically numeric)
            if param == 'concentration':
                # Extract numeric value from the concentration part
                concentration_part = re.findall(r'\d+', value)
                value = int(concentration_part[0]) if concentration_part else None
            elif param == 'replicate':
                # Extract numeric value for replicate
                replicate_part = re.findall(r'\d+', value)
                value = int(replicate_part[0]) if replicate_part else None
            
            # Add extracted value to the result dictionary
            filename_info[param] = value
            
        except (IndexError, ValueError) as e:
            print(f"Error extracting {param} from the filename. {e}")
            filename_info[param] = None  # Use None for missing or unparseable values
    
    # Print extracted information for verification/debugging
    print(filename_info)
    
    return filename_info

# ===========================================
# Read Tapestation Electropherograms
# ===========================================

def read_tapestation(file, ScreenTape='RNA', ladder='electronic ladder', show_plots=True):
    """
    Read and process TapeStation electropherogram data with size calibration.
    
    This function reads TapeStation CSV output files, calibrates molecular sizes
    using a ladder (either electronic or physical), and returns processed data
    with inferred fragment sizes. Size calibration is performed by fitting an
    exponential curve (y = A * e^(kx)) to the ladder peaks.
    
    Parameters
    ----------
    file : str or path-like
        Path to the TapeStation CSV file containing electropherogram data.
        Expected format: columns with 'Well:SampleName' headers.
    ScreenTape : str, optional
        Type of TapeStation assay used. Options are:
        - 'RNA' : Standard RNA ScreenTape (default)
        - 'RNA HS' : High Sensitivity RNA ScreenTape  
        - 'DNA HS 1000' : High Sensitivity DNA ScreenTape (1000bp)
        This determines the electronic ladder peak positions.
    ladder : str or array-like, optional
        Ladder specification for size calibration:
        - 'electronic ladder' : Use built-in electronic ladder for the
          specified ScreenTape method (default)
        - array-like : Custom ladder peak sizes in nucleotides for physical
          ladder runs (not yet fully implemented)
    show_plots : bool, optional
        If True, display diagnostic plots showing:
        - Ladder peak detection
        - Exponential curve fit quality
        Default is True.
    
    Returns
    -------
    df : pandas.DataFrame
        Processed data in long format with columns:
        - 'size (nt)' : Calibrated molecular size in nucleotides
        - 'sample' : Sample name (extracted from original column headers)
        - 'value' : Signal intensity at each position
        - 'screentape' : ScreenTape assay type used
    
    Notes
    -----
    Size calibration algorithm:
    1. Detect peaks in the ladder trace using scipy.signal.find_peaks
    2. Match detected peaks to known ladder sizes
    3. Fit exponential model: size = A * exp(k * position)
       by performing linear regression in log-space
    4. Apply fitted model to all data points to infer sizes
    
    The exponential model assumes that migration distance is logarithmically
    related to molecular size, which is a standard assumption in gel
    electrophoresis.
    
    Currently, only electronic ladders are fully supported. Physical ladder
    support is planned for future versions.
    
    Examples
    --------
    >>> # Read RNA ScreenTape data with default settings
    >>> df = read_tapestation('run_data.csv', ScreenTape='RNA')
    Analysing RNA ScreenTape assay.
    Ladder has peaks at [25, 200, 500, 1000, 2000, 4000, 6000] nucleotides
    
    >>> # Read DNA data without plots
    >>> df = read_tapestation('dna_data.csv', ScreenTape='DNA HS 1000', 
    ...                        show_plots=False)
    
    >>> # Use custom ladder (when implemented)
    >>> custom_ladder = [100, 250, 500, 750, 1000]
    >>> df = read_tapestation('data.csv', ladder=custom_ladder)
    """
    
    # =========================================================================
    # 1. Load and preprocess data
    # =========================================================================
    
    # Read CSV file
    df = pd.read_csv(file)
    
    # Clean column names: remove well IDs (e.g., "A1: Sample" -> "Sample")
    # Expected format: "WellID: SampleName" where WellID is like "A1", "B2", etc.
    df.columns = [col.split(':')[1][1:] for col in df.columns]
    
    # =========================================================================
    # 2. Define ladder peak positions based on assay type
    # =========================================================================
    
    # Use electronic ladder corresponding to ScreenTape method
    if ladder == 'electronic ladder':
        if ScreenTape == 'DNA HS 1000':
            # Standard DNA HS 1000 ladder has 10 peaks spanning 25-1500 nt
            ladder_peaks = [25, 50, 100, 200, 300, 400, 500, 700, 1000, 1500]
            print('Analysing DNA HS 1000 ScreenTape assay.')
            print(f'Ladder has peaks at {ladder_peaks} nucleotides')
        else:
            # Standard RNA ladder has 7 peaks spanning 25-6000 nt
            # Used for both 'RNA' and 'RNA HS' assays
            ladder_peaks = [25, 200, 500, 1000, 2000, 4000, 6000]
            print(f'Analysing {ScreenTape} ScreenTape assay.')
            print(f'Ladder has peaks at {ladder_peaks} nucleotides')
    else:
        # Use custom ladder sizes provided by user
        ladder_peaks = ladder
        print(f'Using custom ladder with peaks at {ladder_peaks} nucleotides')
    
    # =========================================================================
    # 3. Extract ladder trace from data
    # =========================================================================
    
    if ladder == 'electronic ladder':
        # Electronic ladder is stored in a dedicated column
        ladder_values = df['Electronic Ladder']
    else:
        # Physical ladder implementation - to be completed
        # Would need to specify which column contains the ladder run
        print('Actual ladder not yet implemented')
        raise NotImplementedError("Physical ladder support coming soon")
    
    # =========================================================================
    # 4. Detect peaks in ladder trace
    # =========================================================================
    
    # Find local maxima in the ladder signal
    # prominence=100 ensures we only detect significant peaks, not noise
    peaks, _ = find_peaks(ladder_values, prominence=100)
    
    # Verify we found the expected number of peaks
    if len(peaks) != len(ladder_peaks):
        print(f"Warning: Expected {len(ladder_peaks)} peaks but found {len(peaks)}")
    
    # =========================================================================
    # 5. Fit exponential curve to calibrate size vs. position
    # =========================================================================
    # Model: size = A * exp(k * position)
    # Strategy: Transform to linear space using logarithm, then use
    # least squares regression to find parameters
    
    # Transform known sizes to log space: log(size) = log(A) + k * position
    log_y = np.log(ladder_peaks)
    
    # Calculate sums needed for linear regression (least squares solution)
    n = len(peaks)  # number of calibration points
    sum_x = sum(peaks)  # sum of positions
    sum_log_y = sum(log_y)  # sum of log(sizes)
    sum_x_squared = sum(x**2 for x in peaks)  # sum of positions^2
    sum_x_log_y = sum(x * ly for x, ly in zip(peaks, log_y))  # sum of position*log(size)
    
    # Solve for slope (m) and intercept (b) using normal equations
    # These give the least squares solution to: log(size) = b + m * position
    m = (n * sum_x_log_y - sum_x * sum_log_y) / (n * sum_x_squared - sum_x**2)
    b = (sum_log_y - m * sum_x) / n
    
    # Transform back to exponential parameters
    # If log(size) = b + m*position, then size = exp(b) * exp(m*position)
    A = np.exp(b)  # Amplitude/scaling factor
    k = m  # Exponential rate constant
    
    # =========================================================================
    # 6. Apply calibration to all data points
    # =========================================================================
    
    # Calculate size for each position (row index) in the dataframe
    # This converts position/time to molecular size in nucleotides
    df['size (nt)'] = A * np.exp(k * df.index.values)
    
    # =========================================================================
    # 7. Reshape data to long format
    # =========================================================================
    
    # Convert from wide format (one column per sample) to long format
    # This creates one row per (size, sample) combination
    df = df.melt(id_vars='size (nt)', var_name='sample')
    
    # Add metadata column indicating which assay was used
    df['screentape'] = ScreenTape
    
    # =========================================================================
    # 8. Generate diagnostic plots
    # =========================================================================
    
    if show_plots:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        
        # Left plot: Show detected peaks on ladder trace
        ax[0].plot(ladder_values, label='Ladder signal')
        ax[0].plot(peaks, ladder_values[peaks], 'ro', 
                   label=f'Detected peaks (n={len(peaks)})')
        ax[0].set_xlabel('Position (index)')
        ax[0].set_ylabel('Signal intensity')
        ax[0].set_title('Ladder Peak Detection')
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
        
        # Right plot: Show exponential fit quality
        # Plot the calibration points and fitted curve
        x_line = np.linspace(0, peaks.max() + 10, 1000)
        y_line = A * np.exp(k * x_line)
        ax[1].plot(x_line, y_line, label=f'Fit: {A:.2f}*exp({k:.4f}*x)')
        ax[1].plot(peaks, ladder_peaks, 'ro', label='Calibration points')        
        ax[1].set_xlabel('Position (index)')
        ax[1].set_ylabel('Size (nucleotides)')
        ax[1].set_title('Size Calibration Curve')
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return df