import os


def get_notes_from_case(case_type, run_no):
    if 'aec' in case_type:
        logs_dir = '/tmp/rbm_aec_logs'
        net_type = 'autoencoder: '
    else:
        logs_dir = '/tmp/rbm_logs'
        net_type = 'rbm: '
    all_logs = os.listdir(logs_dir)
    log_name = [l for l in all_logs if l.startswith(run_no + '_')][0]
    notes = log_name.strip(run_no + '_')
    return net_type + notes

def binarize_encodings(encodings, threshold=0.2):
    encodings[encodings > threshold] = 1
    encodings[encodings <= threshold] = 0
    encodings = encodings.astype('i4')
    return encodings
