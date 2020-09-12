import logging
import os
import pickle
import tarfile
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.io as sp_io
import shutil
from scipy.sparse import csr_matrix, issparse

from scMVP.dataset.dataset import CellMeasurement, GeneExpressionDataset, _download

logger = logging.getLogger(__name__)
available_datasets = {
        "CellLineMixture": [
            "GSM3271040_RNA_sciCAR_A549_cell.tsv",
            "GSM3271040_RNA_sciCAR_A549_gene.tsv",
            "GSM3271040_RNA_sciCAR_A549_gene.count.mtx",
            "GSM3271041_ATAC_sciCAR_A549_cell.ATAC.tsv",
            "GSM3271041_ATAC_sciCAR_A549_peak.tsv",
            "GSM3271041_ATAC_sciCAR_A549_peak.count.mtx",
        ],
        "only_A549": [
            "GSM3271042_RNA_only_A549_cell.tsv",
            "GSM3271042_RNA_only_A549_gene.tsv",
            "GSM3271042_RNA_only_A549_gene.count.mtx",
            "GSM3271043_ATAC_only_A549_cell.ATAC.tsv",
            "GSM3271043_ATAC_only_A549_peak.tsv",
            "GSM3271043_ATAC_only_A549_peak.count.mtx",
        ],
        "mouse_kidney": [
            "GSM3271044_RNA_mouse_kidney_cell.tsv",
            "GSM3271044_RNA_mouse_kidney_gene.tsv",
            "GSM3271044_RNA_mouse_kidney_gene.count.mtx",
            "GSM3271045_ATAC_mouse_kidney_cell.ATAC.tsv",
            "GSM3271045_ATAC_mouse_kidney_peak.tsv",
            "GSM3271045_ATAC_mouse_kidney_peak.count.mtx",
        ],
    }
available_suffix = {
    "cell.tsv": "gene_barcodes",
    "gene.count.mtx": "gene_expression",
    "gene.tsv": "gene_names",
    "cell.ATAC.tsv":"atac_barcodes",
    "peak.count.mtx":"atac_expression",
    "peak.tsv":"atac_names",
}
available_specification = ["filtered", "raw"]

class scienceDataset(GeneExpressionDataset):
    """Loads a file from `10x`_ website.

    :param dataset_name: Name of the dataset file. Has to be one of:
        "CellLineMixture", "AdBrainCortex", "P0_BrainCortex".
    :param save_path: Location to use when saving/loading the data.
    :param url: manual override of the download remote location.
        Note that we already provide urls for most 10X datasets,
        which are automatically formed only using the ``dataset_name``.
    :param type: Either `filtered` data or `raw` data.
    :param dense: Whether to load as dense or sparse.
        If False, data is cast to sparse using ``scipy.sparse.csr_matrix``.
    :param measurement_names_column: column in which to find measurement names in the corresponding `.tsv` file.
    :param remove_extracted_data: Whether to remove extracted archives after populating the dataset.
    :param delayed_populating: Whether to populate dataset with a delay

    Examples:
        >>> science_daset = scienceDataset("CellLineMixture")


    """


    def __init__(
        self,
        dataset_name: str = None,
        save_path: str = "dataset/",
        type: str = "filtered",
        dense: bool = False,
        measurement_names_column: int = 0,
        remove_extracted_data: bool = False,
        delayed_populating: bool = False,
        datatype = "atac_seq",
        is_binary = False,
    ):
        self.barcodes = None
        self.dense = dense
        self.measurement_names_column = measurement_names_column
        self.remove_extracted_data = remove_extracted_data
        self.datatype = datatype
        self.is_binary = is_binary

        # form data name and filename unless manual override
        self.save_path_list = []
        if dataset_name is not None:
            filenames = available_datasets[dataset_name]
            for filename in filenames:
                self.save_path_list.append(os.path.join(save_path, filename))
        else:
            logger.debug("Loading extracted local snare dataset with custom filename")
        super().__init__()
        if not delayed_populating:
            self.populate()

    def populate(self):
        logger.info("Preprocessing dataset")

        was_extracted = False
        if len(self.save_path_list) > 0:
            for file_path in self.save_path_list:
                if tarfile.is_tarfile(file_path):
                    logger.info("Extracting tar file")
                    tar = tarfile.open(file_path, "r:gz")
                    tar.extractall(path=self.save_path)
                    was_extracted = True
                    tar.close()

        data_dict = {}
        for file_path in self.save_path_list:
            if file_path.split("_")[-1].split(".")[-1] == "mtx" and self.dense:
                data_dict[available_suffix[file_path.split("_")[-1]]] = sp_io.mmread(file_path).T
            elif not self.dense and file_path.split("_")[-1].split(".")[-1] == "mtx":
                data_dict[available_suffix[file_path.split("_")[-1]]] = csr_matrix(sp_io.mmread(file_path).T)
            else:
                if len(self.save_path_list) == 2:
                    data_dict[available_suffix[file_path.split("_")[-1]]] = pd.read_csv(file_path, sep="\t",
                                                                                        header=0, index_col=0)
                else:
                    data_dict[available_suffix[file_path.split("_")[-1]]] = pd.read_csv(file_path, sep=",", header=0)

        if len(self.save_path_list) == 2:
            temp = data_dict["gene_expression"]
            data_dict["gene_barcodes"] = pd.DataFrame(temp.columns.values)
            data_dict["gene_names"] = pd.DataFrame(temp._stat_axis.values)
            data_dict["gene_expression"] = np.array(temp).T

            temp = data_dict["atac_expression"]
            data_dict["atac_barcodes"] = pd.DataFrame(temp.columns.values)
            data_dict["atac_names"] = pd.DataFrame(temp._stat_axis.values)
            data_dict["atac_expression"] = np.array(temp).T
        else:
            temp = data_dict["gene_barcodes"]
            data_dict["gene_barcodes"] = temp['sample']
            if "group" in temp.columns.values.tolist():
                data_dict["gene_label"] = temp['group']
            elif "replicate" in temp.columns.values.tolist():
                data_dict["gene_label"] = temp['replicate']
            elif "cell_name" in temp.columns.values.tolist():
                data_dict["gene_label"] = temp['cell_name']
            else:
                data_dict["gene_label"] = temp['sample']

            temp = data_dict["atac_barcodes"]
            data_dict["atac_barcodes"] = temp['sample']

            if "group" in temp.columns.values.tolist():
                data_dict["atac_label"] = temp['group']
            elif "replicate" in temp.columns.values.tolist():
                data_dict["atac_label"] = temp['replicate']
            elif "cell_name" in temp.columns.values.tolist():
                data_dict["atac_label"] = temp['cell_name']
            else:
                data_dict["atac_label"] = temp['sample']

            if "source" in temp.columns.values.tolist():
                data_dict["atac_source"] = temp['source']

            temp = data_dict["atac_names"]
            data_dict["atac_names"] = temp['peak']




        #gene_barcode_index = np.array(data_dict["gene_barcodes"]).argsort()
        #gene_barcode_index = data_dict["gene_barcodes"].sort_values(by = ["0"],axis = 0).index.tolist()
        xy, gene_barcode_index, atac_barcode_index = np.intersect1d(data_dict["gene_barcodes"].values,
                                                                    data_dict["atac_barcodes"].values,
                                                                    return_indices=True)
        #gene_barcode_index = data_dict["gene_barcodes"].values.tolist()
        #gene_barcode_index = sorted(range(len(gene_barcode_index)),key = lambda k:gene_barcode_index[k])
        #gene_barcode_index = data_dict["gene_barcodes"].values.tolist().index(gene_barcode_index)
        temp = data_dict["gene_barcodes"]
        data_dict["gene_barcodes"] = temp.loc[gene_barcode_index]
        temp = data_dict["gene_expression"]
        if issparse(temp):
            data_dict["gene_expression"] = temp[gene_barcode_index,:].A
        else:
            data_dict["gene_expression"] = temp[gene_barcode_index, :]
        temp = data_dict["gene_label"]
        data_dict["gene_label"] = temp.loc[gene_barcode_index]

        #atac_barcode_index = data_dict["atac_barcodes"].values.tolist()
        #atac_barcode_index = sorted(range(len(atac_barcode_index)), key=lambda k: atac_barcode_index[k])
        temp = data_dict["atac_barcodes"]
        data_dict["atac_barcodes"] = temp.loc[atac_barcode_index]
        temp = data_dict["atac_expression"]
        data_dict["atac_expression"] = temp[atac_barcode_index, :]
        temp = data_dict["atac_label"]
        data_dict["atac_label"] = temp.loc[atac_barcode_index]
        if "atac_source" in data_dict.keys():
            temp = data_dict["atac_source"]
            data_dict["atac_source"] = temp.loc[atac_barcode_index]

        #if issparse(temp):
        #    data_dict["atac_expression"] = temp[atac_barcode_index, :].A
        #else:
        #    data_dict["atac_expression"] = temp[atac_barcode_index, :]
        # filter the atac data
        temp = data_dict["atac_expression"]
        # for binary distribution
        if self.is_binary:
            temp_index = temp > 1
            temp[temp_index] = 1
        # end binary
        high_count_atacs = ((temp > 0).sum(axis=0).ravel() >= 0.001 * temp.shape[0])\
                            & ((temp > 0).sum(axis=0).ravel() <= 0.1 * temp.shape[0])
        #high_count_atacs = ((temp > 0).sum(axis=0).ravel() >= 0.19 * temp.shape[0])



        if issparse(temp):
            high_count_atacs_index = np.where(high_count_atacs)
            temp = temp[:, high_count_atacs_index[1]]
            data_dict["atac_expression"] = temp.A
            data_dict["atac_names"] = data_dict["atac_names"].loc[high_count_atacs_index[1]]
        else:
            temp = temp[:, high_count_atacs]
            data_dict["atac_expression"] = temp
            data_dict["atac_names"] = data_dict["atac_names"].loc[high_count_atacs, :]
            print(len(temp[temp > 1]))
            print(len(temp[temp < 0]))
        #data_dict["atac_expression"] = temp
        #data_dict["atac_names"] = data_dict["atac_names"].loc[high_count_atacs,:]

        '''
        # ATAC-seq as the key
        Ys = []
        measurement = CellMeasurement(
            name="atac_expression",
            data=data_dict["atac_expression"],
            columns_attr_name="atac_names",
            columns=data_dict["atac_names"].astype(np.str),
        )
        Ys.append(measurement)

        cell_attributes_dict = {
            "barcodes": np.squeeze(np.asarray(data_dict["atac_barcodes"], dtype=str))
        }

        logger.info("Finished preprocessing dataset")

        self.populate_from_data(
            X=data_dict["atac_expression"],
            batch_indices=None,
            gene_names=data_dict["atac_names"].astype(np.str),
            cell_attributes_dict=cell_attributes_dict,
            Ys=Ys,
        )
        self.filter_cells_by_count(datatype=self.datatype)
        '''
         # RNA-seq as the key
        label = np.zeros(len(data_dict["atac_label"].values))
        if "atac_label" in data_dict.keys():
            temp = data_dict["atac_label"].values.tolist()
            temp1 = dict(zip(temp, range(len(temp))))
            for i, key in zip(range(len(temp1)),temp1.keys()):
                temp1[key] = i
            for i, el in zip(range(len(temp)),temp):
                label[i] = temp1[el]
            if "atac_source" in data_dict.keys():
                temp2 = data_dict["atac_source"].values.tolist()
                temp3 = dict(zip(temp2, range(len(temp2))))
                for i, key in zip(range(len(temp3)), temp3.keys()):
                    temp3[key] = i+len(temp1)
                for i, el in zip(range(len(temp2)), temp2):
                    label[i] *= temp3[el]
                #temp4 = dict(zip(label, range(len(label))))
                #for i, el in zip(range(len(label)), label):
                    #label[i] = temp4[el]
        elif "gene_label" in data_dict.keys():
            temp = data_dict["gene_label"].values.tolist()
            temp1 = dict(zip(temp, range(len(temp))))
            for i, key in zip(range(len(temp1)),temp1.keys()):
                temp1[key] = i
            for i, el in zip(range(len(temp)), temp):
                label[i] = temp1[el]
        data_dict["label"] = label

        Ys = []
        measurement = CellMeasurement(
            name="atac_expression",
            data=data_dict["atac_expression"],
            columns_attr_name="atac_names",
            columns=data_dict["atac_names"].astype(np.str),
        )
        Ys.append(measurement)

        cell_attributes_dict = {
            "barcodes": np.squeeze(np.asarray(data_dict["gene_barcodes"], dtype=str))
        }

        logger.info("Finished preprocessing dataset")

        self.populate_from_data(
            X=data_dict["gene_expression"],
            batch_indices=None,
            gene_names=data_dict["gene_names"].astype(np.str),
            cell_attributes_dict=cell_attributes_dict,
            Ys=Ys,
            labels = label,
            remap_attributes = False
        )
        self.filter_cells_by_count(datatype = self.datatype)


    '''
        # get exact path of the extract, for robustness to changes is the 10X storage logic
        path_to_data, suffix = self.find_path_to_data()

        # get filenames, according to 10X storage logic
        measurements_filename = "genes.tsv" if suffix == "" else "features.tsv.gz"
        barcode_filename = "barcodes.tsv" + suffix

        matrix_filename = "matrix.mtx" + suffix
        expression_data = sp_io.mmread(os.path.join(path_to_data, matrix_filename)).T
        if self.dense:
            expression_data = expression_data.A
        else:
            expression_data = csr_matrix(expression_data)

        # group measurements by type (e.g gene, protein)
        # in case there are multiple measurements, e.g protein
        # they are indicated in the third column
        gene_expression_data = expression_data
        measurements_info = pd.read_csv(
            os.path.join(path_to_data, measurements_filename), sep="\t", header=None
        )
        Ys = None
        if measurements_info.shape[1] < 3:
            gene_names = measurements_info[self.measurement_names_column].astype(np.str)
        else:
            gene_names = None
            for measurement_type in np.unique(measurements_info[2]):
                # .values required to work with sparse matrices
                measurement_mask = (measurements_info[2] == measurement_type).values
                measurement_data = expression_data[:, measurement_mask]
                measurement_names = measurements_info[self.measurement_names_column][
                    measurement_mask
                ].astype(np.str)
                if measurement_type == "Gene Expression":
                    gene_expression_data = measurement_data
                    gene_names = measurement_names
                else:
                    Ys = [] if Ys is None else Ys
                    if measurement_type == "Antibody Capture":
                        measurement_type = "protein_expression"
                        columns_attr_name = "protein_names"
                        # protein counts do not have many zeros so always make dense
                        if self.dense is not True:
                            measurement_data = measurement_data.A
                    else:
                        measurement_type = measurement_type.lower().replace(" ", "_")
                        columns_attr_name = measurement_type + "_names"
                    measurement = CellMeasurement(
                        name=measurement_type,
                        data=measurement_data,
                        columns_attr_name=columns_attr_name,
                        columns=measurement_names,
                    )
                    Ys.append(measurement)
            if gene_names is None:
                raise ValueError(
                    "When loading measurements, no 'Gene Expression' category was found."
                )

        batch_indices, cell_attributes_dict = None, None
        if os.path.exists(os.path.join(path_to_data, barcode_filename)):
            barcodes = pd.read_csv(
                os.path.join(path_to_data, barcode_filename), sep="\t", header=None
            )
            cell_attributes_dict = {
                "barcodes": np.squeeze(np.asarray(barcodes, dtype=str))
            }
            # As of 07/01, 10X barcodes have format "%s-%d" where the digit is a batch index starting at 1
            batch_indices = np.asarray(
                [barcode.split("-")[-1] for barcode in cell_attributes_dict["barcodes"]]
            )
            batch_indices = batch_indices.astype(np.int64) - 1

        logger.info("Finished preprocessing dataset")

        self.populate_from_data(
            X=gene_expression_data,
            batch_indices=batch_indices,
            gene_names=gene_names,
            cell_attributes_dict=cell_attributes_dict,
            Ys=Ys,
        )
        self.filter_cells_by_count()

        # cleanup if required
        if was_extracted and self.remove_extracted_data:
            logger.info("Removing extracted data at {}".format(file_path[:-7]))
            shutil.rmtree(file_path[:-7])
        '''
    '''
    def find_path_to_data(self) -> Tuple[str, str]:
        """Returns exact path for the data in the archive.

        This is required because 10X doesn't have a consistent way of storing their data.
        Additionally, the function returns whether the data is stored in compressed format.

        :return: path in which files are contains and their suffix if compressed.
        """
        for root, subdirs, files in os.walk(self.save_path):
            # do not consider hidden files
            files = [f for f in files if not f[0] == "."]
            contains_mat = [
                filename == "matrix.mtx" or filename == "matrix.mtx.gz"
                for filename in files
            ]
            contains_mat = np.asarray(contains_mat).any()
            if contains_mat:
                is_tar = files[0][-3:] == ".gz"
                suffix = ".gz" if is_tar else ""
                return root, suffix
        raise FileNotFoundError(
            "No matrix.mtx(.gz) found in path (%s)." % self.save_path
        )
    '''
