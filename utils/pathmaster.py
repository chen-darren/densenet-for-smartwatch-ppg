# -*- coding: utf-8 -*-
"""
Created on Tue Mar 4 13:04:27 2024

@author: dchen
"""
import os

class PathMaster():
    def __init__(self, is_linux=False, is_hpc=False, is_tfs=True, is_internal=False, is_external=False, focus='misc', file_tag='temp', img_res='not_an_img_res'):
        self.focus = focus
        self.file_tag = file_tag
        self.is_linux = is_linux
        self.is_hpc = is_hpc
        self.is_tfs = is_tfs
        self.is_internal = is_internal
        self.is_external = is_external
        self.img_res = img_res
        
        # Set default data and labels paths
        self.data_path, self.labels_path = self.data_paths('pt')
        
        # Select correct root saves path
        if self.is_linux:
            if self.is_tfs:
                self.root_saves_path = '/mnt/R/ENGR_Chon/Darren/Honors_Thesis/saves_tfs'
            else:
                self.root_saves_path = '/mnt/R/ENGR_Chon/Darren/Honors_Thesis/saves_poincare'
        elif self.is_hpc:
            if self.is_tfs:
                self.root_saves_path = '/gpfs/scratchfs1/hfp14002/dac20022/Honors_Thesis/saves_tfs'
            else:
                self.root_saves_path = '/gpfs/scratchfs1/hfp14002/dac20022/Honors_Thesis/saves_poincare'
        else: # Using your own computer
            if self.is_tfs:
                self.root_saves_path = r'\\grove.ad.uconn.edu\research\ENGR_Chon\Darren\Honors_Thesis\saves_tfs'
            else:
                self.root_saves_path = r'\\grove.ad.uconn.edu\research\ENGR_Chon\Darren\Honors_Thesis\saves_poincare'
        self.saves_path = os.path.join(self.root_saves_path, self.focus)


    def set_saves_path(self, saves_path):
        self.saves_path = saves_path


    def set_file_tag(self, file_tag):
        self.file_tag = file_tag
        
        
    def set_focus(self, focus):
        self.focus = focus

    def set_data_labels_paths(self, data_path, labels_path):
        self.data_path = data_path
        self.labels_path = labels_path
        
    def data_paths(self, data_format):
        if data_format == 'pt':
            # Base path
            if self.is_linux:
                data_root_path = "/mnt/R/ENGR_Chon/Darren/NIH_PulseWatch"
                labels_root_path = "/mnt/R/ENGR_Chon/Darren/NIH_Pulsewatch"
                # labels_root_path = "/mnt/R/ENGR_Chon/NIH_Pulsewatch_Database/Adjudication_UConn"
            elif self.is_hpc:
                data_root_path = "/gpfs/scratchfs1/kic14002/doh16101"
                labels_root_path = "/gpfs/scratchfs1/hfp14002/lrm22005"
            else:
                if self.is_internal:
                    data_root_path = r'C:\Chon_Lab\NIH_Pulsewatch'
                    labels_root_path = r'C:\Chon_Lab\NIH_Pulsewatch'
                elif self.is_external:
                    data_root_path = r'D:\Chon_Lab\NIH_Pulsewatch'
                    labels_root_path = r'D:\Chon_Lab\NIH_Pulsewatch'
                else:
                    # R:\ENGR_Chon\Dong\MATLAB_generate_results\NIH_PulseWatch
                    data_root_path = "R:\ENGR_Chon\Darren\\NIH_Pulsewatch" # Why double \\ before NIH_Pulsewatch_Database?
                    labels_root_path = "R:\ENGR_Chon\Darren\\NIH_Pulsewatch" # Why double \\ before NIH_Pulsewatch_Database?      
                    # labels_root_path = "R:\ENGR_Chon\\NIH_Pulsewatch_Database\Adjudication_UConn"      
            
            # Type path
            if self.is_tfs:
                format_path = 'TFS_pt'
            else:
                format_path = 'Poincare_pt'
            
            # Join paths
            data_path = os.path.join(data_root_path, format_path, self.img_res)
            
        else:
            if self.is_linux:
                data_root_path = "/mnt/R/ENGR_Chon/Dong/MATLAB_generate_results/NIH_PulseWatch"
                labels_root_path = "/mnt/R/ENGR_Chon/Darren/NIH_Pulsewatch"
                # labels_root_path = "/mnt/R/ENGR_Chon/NIH_Pulsewatch_Database/Adjudication_UConn"
            elif self.is_hpc:
                data_root_path = "/gpfs/scratchfs1/kic14002/doh16101"
                labels_root_path = "/gpfs/scratchfs1/hfp14002/lrm22005"
            else:
                # R:\ENGR_Chon\Dong\MATLAB_generate_results\NIH_PulseWatch
                data_root_path = "R:\ENGR_Chon\Dong\MATLAB_generate_results\\NIH_PulseWatch" # Why double \\ before NIH_Pulsewatch_Database?
                labels_root_path = "R:\ENGR_Chon\Darren\\NIH_Pulsewatch" # Why double \\ before NIH_Pulsewatch_Database?
                # labels_root_path = "R:\ENGR_Chon\\NIH_Pulsewatch_Database\Adjudication_UConn"
            
            if data_format == 'csv':
                if self.is_tfs:
                    data_path = os.path.join(data_root_path, "TFS_csv")
                else:
                    data_path = os.path.join(data_root_path, "Poincare_Density_csv")
            elif data_format == 'png':
                if not self.is_tfs:
                    print('No png image available for Density Poincare plot')
                    return
                data_path = os.path.join(data_root_path, "TFS_plots")
            else:
                raise ValueError("Invalid data format. Choose 'csv', 'png, or 'pt'.")
        
        # Complete labels path
        # labels_path = os.path.join(labels_root_path, "final_attemp_4_1_Dong_Ohm_2024_02_18_copy")
        labels_path = os.path.join(labels_root_path, "Ground_Truths")
        
        # Check if directories exist        
        if not os.path.exists(data_path):
            print("Data path does not exist")
            return    
        if not os.path.exists(labels_path):
            print("Labels path does not exist")
            return          

        return data_path, labels_path
    
    
    def combination_path(self, combination, split):
        if self.is_internal:
            root_path = r'C:\Chon_Lab\NIH_Pulsewatch'
        elif self.is_external:
            root_path = r'D:\Chon_Lab\NIH_Pulsewatch'
        else:
            # R:\ENGR_Chon\Dong\MATLAB_generate_results\NIH_PulseWatch
            root_path = "R:\ENGR_Chon\Darren\\NIH_Pulsewatch" # Why double \\ before NIH_Pulsewatch_Database?

        # Type path
        if self.is_tfs:
            format_path = 'TFS_pt'
        else:
            format_path = 'Poincare_pt'
        
        combination_path = os.path.join(root_path, format_path, combination, split)
        
        return combination_path
    
    
    def deepbeat_paths(self):
        if self.is_internal:
            root_path = r'C:\Chon_Lab\Public_Database\DeepBeat\Concatenated_DeepBeat\test\Darren_conversion'
        elif self.is_external:
            root_path = r'D:\Chon_Lab\Public_Database\DeepBeat\Concatenated_DeepBeat\test\Darren_conversion'
        else:
            # R:\ENGR_Chon\Dong\MATLAB_generate_results\NIH_PulseWatch
            root_path = r'R:\ENGR_Chon\Darren\Public_Database\DeepBeat\Concatenated_DeepBeat\test\Darren_conversion'

        # Type path
        if self.is_tfs:
            format_path = 'tfs_float16_pt'
        else:
            format_path = 'poincare_float16_pt'
        
        data_path = os.path.join(root_path, format_path)
        labels_path = os.path.join(root_path, 'DeepBeat_segment_names_labels_STFT.csv')
        
        return data_path, labels_path
    
    
    def mimic3_paths(self):
        # Old version
        if self.is_internal:
            root_path = r'C:\Chon_Lab\Public_Database\PPG_PeakDet_MIMICIII\Darren_conversion'
        elif self.is_external:
            root_path = r'D:\Chon_Lab\Public_Database\PPG_PeakDet_MIMICIII\Darren_conversion'
        else:
            # R:\ENGR_Chon\Dong\MATLAB_generate_results\NIH_PulseWatch
            root_path = r'R:\ENGR_Chon\Darren\Public_Database\PPG_PeakDet_MIMICIII\Darren_conversion'

        # Type path
        if self.is_tfs:
            format_path = 'test_tfs_float16_pt'
        else:
            format_path = 'test_poincare_float16_pt'
        
        data_path = os.path.join(root_path, format_path)
        labels_path = os.path.join(root_path, '2020_Han_Sensors_MIMICIII_Ground_Truth_STFT.csv')
        
        # # New version
        # if self.is_internal:
        #     root_path = r'C:\Chon_Lab\Public_Database\MIMIC_III'
        # elif self.is_external:
        #     root_path = r'D:\Chon_Lab\Public_Database\MIMIC_III'
        # else:
        #     # R:\ENGR_Chon\Dong\MATLAB_generate_results\NIH_PulseWatch
        #     root_path = r'R:\ENGR_Chon\Darren\Public_Database\MIMIC_III'

        # # Type path
        # if self.is_tfs:
        #     format_path = os.path.join('TFS_pt', '128x128_float16')
        # else:
        #     format_path = os.path.join('Poincare_pt', '128x128_float16')
        
        # data_path = os.path.join(root_path, format_path)
        # labels_path = os.path.join(root_path, 'Ground_Truths')
        
        return data_path, labels_path
    
    
    def simband_paths(self):
        # Old version
        if self.is_internal:
            root_path = r'C:\Chon_Lab\Public_Database\PPG_PeakDet_Simband\Darren_conversion'
        elif self.is_external:
            root_path = r'D:\Chon_Lab\Public_Database\PPG_PeakDet_Simband\Darren_conversion'
        else:
            # R:\ENGR_Chon\Dong\MATLAB_generate_results\NIH_PulseWatch
            root_path = r'R:\ENGR_Chon\Darren\Public_Database\PPG_PeakDet_Simband\Darren_conversion'

        # Type path
        if self.is_tfs:
            format_path = 'tfs_float16_pt'
        else:
            format_path = 'poincare_float16_pt'
        
        data_path = os.path.join(root_path, format_path)
        labels_path = os.path.join(root_path, 'simband_segments_labels_STFT.csv')
        
        # # New version
        # if self.is_internal:
        #     root_path = r'C:\Chon_Lab\Public_Database\Simband'
        # elif self.is_external:
        #     root_path = r'D:\Chon_Lab\Public_Database\Simband'
        # else:
        #     # R:\ENGR_Chon\Dong\MATLAB_generate_results\NIH_PulseWatch
        #     root_path = r'R:\ENGR_Chon\Darren\Public_Database\Simband'

        # # Type path
        # if self.is_tfs:
        #     format_path = os.path.join('TFS_pt', '128x128_float16')
        # else:
        #     format_path = os.path.join('Poincare_pt', '128x128_float16')
        
        # data_path = os.path.join(root_path, format_path)
        # labels_path = os.path.join(root_path, 'Ground_Truths')
        
        return data_path, labels_path
    

    def summary_path(self):
        if self.is_linux:
            summary_path = "/mnt/R/ENGR_Chon/Darren/NIH_Pulsewatch/labels_summary_2_18_Darren.csv"
        elif self.is_hpc:
            summary_path = "/gpfs/scratchfs1/hfp14002/dac20022/NIH_Pulsewatch/labels_summary_2_18_Darren.csv"
        else:
            if self.is_internal:
                summary_path = r'C:\Chon_Lab\NIH_Pulsewatch\labels_summary_2_18_Darren.csv'
            elif self.is_external:
                summary_path = r'D:\Chon_Lab\NIH_Pulsewatch\labels_summary_2_18_Darren.csv'
            else:
                summary_path = r"\\grove.ad.uconn.edu\research\ENGR_Chon\Darren\NIH_Pulsewatch\labels_summary_2_18_Darren.csv"
            
        return summary_path
     

    def models_path(self):
        if self.is_linux:
            models_path = "/mnt/R/ENGR_Chon/Darren/Honors_Thesis/models"
        elif self.is_hpc:
            models_path = "/gpfs/scratchfs1/hfp14002/dac20022/Honors_Thesis/models"
        else:
            models_path = r"\\grove.ad.uconn.edu\research\ENGR_Chon\Darren\Honors_Thesis\models"
        
        return models_path


    def losslists_path(self):
        losslists_path = os.path.join(self.saves_path, 'losslists')
        
        return losslists_path


    def runtime_lists_path(self):
        runtime_lists_path = os.path.join(self.saves_path, 'runtime_lists')
        
        return runtime_lists_path


    def ground_truths_path(self):
        ground_truths_path = os.path.join(self.saves_path, 'labels')
        
        return ground_truths_path


    def predictions_path(self):
        predictions_path = os.path.join(self.saves_path, 'predictions')
        
        return predictions_path


    def prediction_proba_path(self):
        prediction_proba_path = os.path.join(self.saves_path, 'prediction_proba')
        
        return prediction_proba_path


    def metrics_path(self):
        metrics_path = os.path.join(self.saves_path, 'metrics')
    
        return metrics_path
    
    
    def classification_report_path(self):
        classification_report_path = os.path.join(self.saves_path, 'classification_reports')
    
        return classification_report_path
    
    
    def classification_report_imbalanced_path(self):
        classification_report_imbalanced_path = os.path.join(self.saves_path, 'classification_reports_imbalanced')
    
        return classification_report_imbalanced_path


    def confusion_matrices_path(self):
        confusion_matrices_path = os.path.join(self.saves_path, 'confusion_matrices')
    
        return confusion_matrices_path


    def checkpoints_path(self):
        checkpoints_path = os.path.join(self.saves_path, 'checkpoints')

        return checkpoints_path


    def hyperparameters_path(self):
        hyperparameters_path = os.path.join(self.saves_path, 'hyperparameters')
    
        return hyperparameters_path


    def loss_curves_path(self):
        loss_curves_path = os.path.join(self.saves_path, 'loss_curves')
    
        return loss_curves_path
    
    
    def roc_curves_path(self):
        roc_curves_path = os.path.join(self.saves_path, 'roc_curves')
    
        return roc_curves_path
    
    
    def mean_roc_curves_path(self):
        mean_roc_curves_path = os.path.join(self.saves_path, 'mean_roc_curves')
    
        return mean_roc_curves_path
    
    
    def accuracy_curves_path(self):
        accuracy_curves_path = os.path.join(self.saves_path, 'accuracy_curves')
    
        return accuracy_curves_path
    
    def segment_names_path(self):
        segment_names_path = os.path.join(self.saves_path, 'segment_names')
    
        return segment_names_path
    
    def output_file_path(self):
        output_file_path = os.path.join(self.saves_path, 'output_files')
    
        return output_file_path