import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, Markdown
import re

# Load Datasets
DATASETS_PATH = '/gpfs/workdir/restrepoda/datasets/'
DATASETS = ['VLMed', 'mBRSET', 'BRSET', 'MIMIC', 'HAM10000']


def load_medeval(train=True, validation=True, path=os.path.join(DATASETS_PATH, DATASETS[0]), images_path='Dataset', metadata_path='ReadMe/data_summary.csv'):
    """
    Load the VLMed dataset.
    
    Parameters
    ----------
    train : bool
        Load the training set.
    validation : bool
        Load the validation set.
    path : str
        Path to the dataset.
    images_path : str
        Path to the images.
    metadata_path : str
        Path to the metadata.
        
    Returns
    -------
    metadata_train : pd.DataFrame
        Metadata of the training set.
    metadata_val : pd.DataFrame
        Metadata of the validation set.
    metadata_test : pd.DataFrame
        Metadata of the test set.
    """
    
    print(f"Loading VLMed dataset from {path}")
    
    images_path = os.path.join(path, images_path)
    print(f"Images are stored in {os.path.join(path, 'Dataset')}/['Test', 'Validation', or 'Training']")
    
    metadata_path = os.path.join(path, metadata_path)
    metadata = pd.read_csv(metadata_path)
    
    # set filepath for each image
    def get_image_path(row):
        if row['use'] == 'test':
            return os.path.join(images_path, 'Test', row['filename'])
        if row['use'] == 'validation':
            return os.path.join(images_path, 'Validation', row['filename'])
        if row['use'] == 'training':
            return os.path.join(images_path, 'Training', row['filename'])
    
    metadata['filepath'] = metadata.apply(get_image_path, axis=1)
    
    # Split metadata
    metadata_test = metadata[metadata.use == 'test']
    if train:
        metadata_train = metadata[metadata.use == 'training']
    if validation:
        metadata_val = metadata[metadata.use == 'validation']
        
    if train and validation:
        return metadata_train, metadata_val, metadata_test
    elif train:
        return metadata_train, metadata_test
    elif validation:
        return metadata_val, metadata_test
    else:
        return metadata_test
    
import zipfile

def read_text_from_zip(zip_path, file_path):
    """
    Read text from a zip file.
    
    Parameters
    ----------
    zip_path : str
        Path to the zip file.
    file_path : str
        Path to the file inside the zip file.
        
    Returns
    -------
    str
        Text from the file.
    """
    with zipfile.ZipFile(zip_path) as z:
        with z.open(file_path) as f:
            return f.read().decode('utf-8')
        

def extract_texts_from_zip(zip_path, file_list):
    """
    Efficiently extracts multiple text files from a zip archive.
    
    Parameters
    ----------
    zip_path : str
        Path to the zip file.
    file_list : list
        List of file paths inside the zip archive to extract.
        
    Returns
    -------
    dict
        Dictionary with file paths as keys and extracted text as values.
    """
    file_contents = {}
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        existing_files = set(z.namelist())  # Convert file list in ZIP to a set for fast lookup
        
        for file_path in file_list:
            if file_path in existing_files:  # Check if the file exists in the ZIP
                with z.open(file_path) as f:
                    file_contents[file_path] = f.read().decode('utf-8')
            else:
                file_contents[file_path] = None  # If file not found, return None
    
    return file_contents




def convert_to_multiclass(df, label_columns=[
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax'
], label_pneumonia=False, label_pl_effusion=False):
    """
    Convert multi-label classification to 4-class classification:
    'No Finding', 'Pleural Effusion', 'Pneumonia', and 'Others'
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the labels
    label_columns : list
        List of label columns
        
    Returns
    -------
    pd.DataFrame
        DataFrame with new column 'class_label' for multi-class classification
    """
    df = df.copy()
    
    # First check if all labels are 0 (including after -1 conversion)
    all_zeros = (df[label_columns] == 0).all(axis=1)
    
    # Create the new class assignments
    def assign_class(row, label_pneumonia=False, label_pl_effusion=False):
        if all_zeros[row.name]:
            return 'Undefined'
        elif row['No Finding'] == 1:
            return 'No Finding'
        elif label_pl_effusion and row['Pleural Effusion'] == 1:
            return 'Pleural Effusion'
        elif not label_pl_effusion and row['Pleural Effusion'] == 1:
            return 'Others'
        elif label_pneumonia and row['Pneumonia'] == 1:
            return 'Pneumonia'
        elif not label_pneumonia and row['Pneumonia'] == 1:
            return 'Others'
        elif any(row[col] == 1 for col in label_columns if col not in ['No Finding', 'Pleural Effusion', 'Pneumonia']):
            return 'Others'
        else:
            return 'Undefined'
    
    # Add new column with class labels
    df['class_label'] = df.apply(assign_class, axis=1, label_pneumonia=label_pneumonia, label_pl_effusion=label_pl_effusion)
    
    # Remove rows with 'Undefined' class
    df = df[df['class_label'] != 'Undefined']
    
    # If is binary classification classify as normal or abnormal
    if not label_pneumonia and not label_pl_effusion:
        #df['class_label'] = df['class_label'].apply(lambda x: 'Normal' if x == 'No Finding' else 'Abnormal')
        df['class_label'] = df['class_label'].apply(lambda x: 'No' if x == 'No Finding' else 'Yes')
    return df
    
    
def remove_uncertain_labels(df, label_columns=[
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax'
], label_pneumonia=False, label_pl_effusion=False):
    """
    Remove uncertain labels from the dataset and convert to multi-class or binary.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the labels.
    label_columns : list
        List of label columns.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with the uncertain labels removed and multi-class conversion.
    """
    # Handle uncertain labels
    for column in label_columns:
        df[column] = df[column].apply(lambda x: 0.0 if x == -1.0 else x)
    
    # Convert to binary classification
    df = convert_to_multiclass(df, label_columns, label_pneumonia, label_pl_effusion)
    
    return df

    
def load_mimic(train=True, validation=True, path=os.path.join(DATASETS_PATH, DATASETS[3]), images_path='mimic', metadata_path='mimic/', check_images=False, label_pneumonia=False, label_pl_effusion=False):
    
    print(f"Loading MIMIC dataset from {path}")
    
    #disease_df = pd.read_csv(os.path.join(path, metadata_path, 'mimic-cxr-2.0.0-chexpert.csv'))  
    text_path = pd.read_csv(os.path.join(path, metadata_path, 'cxr-study-list.csv'))
    text_path.rename(columns={'path': 'text_path'}, inplace=True)
    
    print("Reading test data...")
    test_df = pd.read_csv(os.path.join(path, metadata_path, 'test.csv'), index_col=0)
    #test_df = test_df.merge(disease_df, on=['subject_id', 'study_id'], how='inner')
    test_df = test_df.merge(text_path, on=['subject_id', 'study_id'], how='inner')
    
    if check_images:
        # Check if the images are available in 'path_preproc'
        print("Checking if images are available for test...")
        for image in test_df['path_preproc']:
            if not os.path.exists(os.path.join(path, images_path, image)):
                print(f"Image not found: {image}")
                
    
    test_df['filepath'] = test_df['path_preproc'].apply(lambda x: os.path.join(path, images_path, x))
    
    print("Reading text from zip file...")
    # Add text from text_path. 
    #test_df['text'] = test_df['text_path'].apply(lambda x: read_text_from_zip(os.path.join(path, metadata_path, 'metadata', 'mimic-cxr-reports.zip'), x))
    zip_file_path = os.path.join(path, metadata_path, 'metadata', 'mimic-cxr-reports.zip')
    
    test_texts = extract_texts_from_zip(zip_file_path, test_df['text_path'].tolist())
    test_df['report'] = test_df['text_path'].map(test_texts)

    # remove uncertain labels
    test_df = remove_uncertain_labels(test_df, label_pneumonia=label_pneumonia, label_pl_effusion=label_pl_effusion)
    

    if train:
        print("Reading train data...")
        train_df = pd.read_csv(os.path.join(path, metadata_path, 'train.csv'), index_col=0)
        #train_df = train_df.merge(disease_df, on=['subject_id', 'study_id'], how='inner')
        train_df = train_df.merge(text_path, on=['subject_id', 'study_id'], how='inner')
                
        if check_images:
            # Check if the images are available in 'path_preproc'
            print("Checking if images are available for train...")
            for image in train_df['path_preproc']:
                if not os.path.exists(os.path.join(path, images_path, image)):
                    print(f"Image not found: {image}")
        
        train_df['filepath'] = train_df['path_preproc'].apply(lambda x: os.path.join(path, images_path, x))
        print("Reading text from zip file...")
        #train_df['text'] = train_df['text_path'].apply(lambda x: read_text_from_zip(os.path.join(path, metadata_path, 'metadata', 'mimic-cxr-reports.zip'), x))
        train_texts = extract_texts_from_zip(zip_file_path, train_df['text_path'].tolist())
        train_df['report'] = train_df['text_path'].map(train_texts)
        
        # remove uncertain labels
        train_df = remove_uncertain_labels(train_df, label_pneumonia=label_pneumonia, label_pl_effusion=label_pl_effusion)
        
    if validation:
        print("Reading validation data...")
        valid_df = pd.read_csv(os.path.join(path, metadata_path, 'valid.csv'), index_col=0)
        #valid_df = valid_df.merge(disease_df, on=['subject_id', 'study_id'], how='inner')
        valid_df = valid_df.merge(text_path, on=['subject_id', 'study_id'], how='inner')
        
        if check_images:
            # Check if the images are available in 'path_preproc'
            print("Checking if images are available for validation...")
            for image in valid_df['path_preproc']:
                if not os.path.exists(os.path.join(path, images_path, image)):
                    print(f"Image not found: {image}")
                
        valid_df['filepath'] = valid_df['path_preproc'].apply(lambda x: os.path.join(path, images_path, x))
        print("Reading text from zip file...")
        #valid_df['text'] = valid_df['text_path'].apply(lambda x: read_text_from_zip(os.path.join(path, metadata_path, 'metadata', 'mimic-cxr-reports.zip'), x))
        valid_texts = extract_texts_from_zip(zip_file_path, valid_df['text_path'].tolist())
        valid_df['report'] = valid_df['text_path'].map(valid_texts)
        
        # remove uncertain labels
        valid_df = remove_uncertain_labels(valid_df, label_pneumonia=label_pneumonia, label_pl_effusion=label_pl_effusion)
    
    if train and validation:
        return train_df, valid_df, test_df
    elif train:
        return train_df, test_df
    elif validation:
        return valid_df, test_df
    else:
        return test_df
    

    
    
def load_ham10000(train=True, validation=True, path=os.path.join(DATASETS_PATH, DATASETS[4]), images_path='images', metadata_path='HAM10000_metadata.tab', check_images=False, seed=42, test_size=0.2, valid_size=0.2, classes=None):
    """
    Load the HAM10000 dataset.
    
    Parameters
    ----------
    train : bool
        Load the training set.
    validation : bool
        Load the validation set.
    path : str
        Path to the dataset.
    images_path : str
        Path to the images.
    metadata_path : str
        Path to the metadata.
        
    Returns
    -------
    metadata_train : pd.DataFrame
        Metadata of the training set.
    metadata_val : pd.DataFrame
        Metadata of the validation set.
    metadata_test : pd.DataFrame
        Metadata of the test set.
    """
    
    df = pd.read_csv(os.path.join(path, metadata_path))
    df['filepath'] = df['image_id'].apply(lambda x: os.path.join(path, images_path, f"{x}.jpg"))
    
    if classes:
        if classes == 2:
            # Only keep values with the 2 most common classes
            df = df[df['dx'].isin(['nv', 'mel'])]
        elif classes == 3:
            # Only keep values with the 3 most common classes
            df = df[df['dx'].isin(['nv', 'mel', 'bkl'])]
        elif classes == 4:
            # Only keep values with the 4 most common classes
            df = df[df['dx'].isin(['nv', 'mel', 'bkl', 'bcc'])]
        elif classes == 5:
            df = df[df['dx'].isin(['nv', 'mel', 'bkl', 'bcc', 'akiec'])]
        elif classes == 6:
            df = df[df['dx'].isin(['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc'])]
        else:
            df = df
    
    if check_images:
        # Check if the images are available in 'path_preproc'
        print("Checking if images are available...")
        for image in df['filepath']:
            if not os.path.exists(image):
                print(f"Image not found: {image}")
    
    # if train and validation, split the data using a seed for reproducibility
    if train and validation:
        print(f"Using test size of {test_size} and validation size of {valid_size} from the training set")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
        train_df, valid_df = train_test_split(train_df, test_size=valid_size, random_state=seed)
        return train_df, valid_df, test_df
    elif train:
        print(f"Using test size of {test_size} from the training set")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
        return train_df, test_df
    elif validation:
        print(f"Using validation size of {valid_size}")
        valid_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
        return valid_df, test_df
    else:
        print("Using the full dataset as the test set")
        return df
    
    
def load_mbrset(train=True, validation=True, path=os.path.join(DATASETS_PATH, DATASETS[1]), images_path='mbrset/images', metadata_path='mbrset/labels_mbrset.csv', check_images=False, seed=42, test_size=0.2, valid_size=0.2, classes=None):
    """
    Load the mBRSET dataset.
    
    Parameters
    ----------
    train : bool
        Load the training set.
    validation : bool
        Load the validation set.
    path : str
        Path to the dataset.
    images_path : str
        Path to the images.
    metadata_path : str
        Path to the metadata.
        
    Returns
    -------
    metadata_train : pd.DataFrame
        Metadata of the training set.
    metadata_val : pd.DataFrame
        Metadata of the validation set.
    metadata_test : pd.DataFrame
        Metadata of the test set.
    """
                                                               
    df = pd.read_csv(os.path.join(path, metadata_path))
    
    # Remove rows with missing values in the lable final_icdr
    df = df.dropna(subset=['final_icdr'])
    # Make the task a binary classification
    df['final_icdr'] = df['final_icdr'].apply(lambda x: 'No' if x == 0 else 'Yes')
    
    # Add path to images
    df['filepath'] = df['file'].apply(lambda x: os.path.join(path, images_path, f"{x}"))
    # If verify_images is True, check if the images are available
    if check_images:
        print("Checking if images are available...")
        for image in df['filepath']:
            if not os.path.exists(image):
                print(f"Image not found: {image}")
                
    # if train and validation, split the data using a seed for reproducibility
    if train and validation:
        print(f"Using test size of {test_size} and validation size of {valid_size} from the training set")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
        train_df, valid_df = train_test_split(train_df, test_size=valid_size, random_state=seed)
        return train_df, valid_df, test_df
    elif train:
        print(f"Using test size of {test_size} from the training set")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
        return train_df, test_df
    elif validation:
        print(f"Using validation size of {valid_size}")
        valid_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
        return valid_df, test_df
    else:
        print("Using the full dataset as the test set")
        return df
    
    
def load_brset(train=True, validation=True, path=os.path.join(DATASETS_PATH, DATASETS[2]), images_path='brset/images', metadata_path='brset/labels_brset.csv', check_images=False, seed=42, test_size=0.2, valid_size=0.2, classes=None):
    """
    Load the BRSET dataset.
    
    Parameters
    ----------
    train : bool
        Load the training set.
    validation : bool
        Load the validation set.
    path : str
        Path to the dataset.
    images_path : str
        Path to the images.
    metadata_path : str
        Path to the metadata.
        
    Returns
    -------
    metadata_train : pd.DataFrame
        Metadata of the training set.
    metadata_val : pd.DataFrame
        Metadata of the validation set.
    metadata_test : pd.DataFrame
        Metadata of the test set.
    """
                                                               
    df = pd.read_csv(os.path.join(path, metadata_path))
    #print(df.columns)
    
    # Remove rows with missing values in the lable final_icdr
    df = df.dropna(subset=['DR_ICDR'])
    # Make the task a binary classification
    df['DR'] = df['DR_ICDR'].apply(lambda x: 'No' if x == 0 else 'Yes')
    
    # Add path to images
    df['filepath'] = df['image_id'].apply(lambda x: os.path.join(path, images_path, f"{x}.jpg"))
    # If verify_images is True, check if the images are available
    if check_images:
        print("Checking if images are available...")
        for image in df['filepath']:
            if not os.path.exists(image):
                print(f"Image not found: {image}")
                
    # if train and validation, split the data using a seed for reproducibility
    if train and validation:
        print(f"Using test size of {test_size} and validation size of {valid_size} from the training set")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
        train_df, valid_df = train_test_split(train_df, test_size=valid_size, random_state=seed)
        return train_df, valid_df, test_df
    elif train:
        print(f"Using test size of {test_size} from the training set")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
        return train_df, test_df
    elif validation:
        print(f"Using validation size of {valid_size}")
        valid_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
        return valid_df, test_df
    else:
        print("Using the full dataset as the test set")
        return df
    
    
    

def plot_image_and_report(df, title=None, n_images=5, random_state=42,
                          metadata_columns=['dicom_id', 'subject_id', 'StudyDate',
                                            'PerformedProcedureStepDescription', 'report',
                                            'class_label']):
    """
    Display a random sample of images with corresponding report metadata.
    """
    if title:
        display(Markdown(f"## {title}"))
    sample_df = df.sample(n=n_images, random_state=random_state)
    for _, row in sample_df.iterrows():
        # Display metadata
        md_text = "### Sample Metadata\n"
        for col in metadata_columns:
            md_text += f"**{col}:** {row.get(col, '')}  \n"
        display(Markdown(md_text))
        # Display image
        img_path = row.get('filepath', '')
        if os.path.exists(img_path):
            try:
                image = Image.open(img_path)
                plt.figure()
                plt.imshow(image, cmap='gray')
                plt.axis('off')
                plt.title(f"Image for dicom_id: {row.get('dicom_id', '')}")
                plt.show()
            except Exception as e:
                print(f"Error loading image from {img_path}: {e}")
        else:
            print(f"Image file not found: {img_path}")


def clean_mimic_reports(path="/gpfs/workdir/restrepoda/datasets/MIMIC/mimic/",
                        filename="test_preproc.csv",
                        plot=True, n_images=5, random_state=42,
                        save_csv=True, out_filename="test_preproc.csv"):
    """
    Load MIMIC report CSV, optionally plot samples, clean report structure,
    extract sections, and return a dataframe of fully structured reports.
    Optionally save the cleaned DataFrame to CSV.
    """
    # Read data
    file_path = os.path.join(path, filename)
    df = pd.read_csv(file_path, index_col=0)
    
    # Remove rows with missing values in the report
    df = df.dropna(subset=['report'])
    # Remove rows with empty reports
    df = df[df['report'].str.strip() != '']

    # Optional plotting of raw samples
    if plot:
        plot_image_and_report(df, title="Original MIMIC Reports",
                              n_images=n_images, random_state=random_state)

    # Define header normalization
    ALL_HEADERS = [
        "INDICATION",
        "TECHNIQUE",
        "COMPARISON",
        "FINDINGS",
        "IMPRESSION"
    ]
    # Aliases mapping to canonical headers
    ALIASES = {
        r"\bREASON\s+FOR\s+EXAMINATION?\b": "INDICATION",
        r"\bCLINICAL\s+INFORMATION\b": "INDICATION",
        r"\bEXAM(INATION)?\b": "TECHNIQUE",
    }

    def normalize_text(text: str) -> str:
        t = text.upper() if isinstance(text, str) else ''
        t = re.sub(r"[ 	]+", " ", t)
        for pat, repl in ALIASES.items():
            t = re.sub(pat, repl, t)
        return t

    # Apply normalization
    df['report_norm'] = df['report'].apply(normalize_text)

    # Filter structured reports
    def has_all_headers(text: str) -> bool:
        return all(re.search(rf"\b{hdr}\s*:", text) for hdr in ALL_HEADERS)

    structured = df[df['report_norm'].apply(has_all_headers)].copy()

    # Extract section content
    section_patterns = {}
    # Build regex pattern to capture each section
    hdr_regex = '|'.join([re.escape(h) + r"\s*:" for h in ALL_HEADERS])
    for i, hdr in enumerate(ALL_HEADERS):
        # Determine lookahead: next headers or end of string
        if i < len(ALL_HEADERS) - 1:
            next_hdr = ALL_HEADERS[i+1]
            pattern = rf"{hdr}\s*:(.*?)(?={next_hdr}\s*:|$)"
        else:
            pattern = rf"{hdr}\s*:(.*)$"
        section_patterns[hdr.lower()] = re.compile(pattern, re.IGNORECASE | re.DOTALL)

    # Extract and clean up whitespace
    for col, pat in section_patterns.items():
        structured[col] = structured['report_norm'].apply(
            lambda text: pat.search(text).group(1).strip() if pat.search(text) else '')

    # Build clean DataFrame with selected columns
    #keep_cols = ['dicom_id', 'subject_id', 'StudyDate', 'PerformedProcedureStepDescription',
    #             'class_label', 'filepath'] + list(section_patterns.keys())
    # structured = structured[keep_cols]
    # Drop original report and normalized report columns
    clean_df = structured.rename(columns={
        'PerformedProcedureStepDescription': 'procedure',
        'class_label': 'label'
    })

    if plot:
        # Plot cleaned samples
        plot_image_and_report(clean_df, title="Cleaned MIMIC Reports",
                              n_images=n_images, random_state=random_state)
    
    # Optionally save to CSV
    if save_csv:
        out_path = os.path.join(path, out_filename)
        clean_df.to_csv(out_path, index=False)
        print(f"Cleaned data saved to {out_path}")

    return clean_df