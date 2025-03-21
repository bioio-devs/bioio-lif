from enum import Enum, unique


@unique
class MetadataLabels(Enum):
    BINNING = "Binning"
    COLUMN = "Column"
    IMAGED_BY = "Imaged_By"
    IMAGING_DATE = "Imaging_Date"
    OBJECTIVE = "Objective"
    POSITION_INDEX = "Position_Index"
    ROW = "Row"
    TOTAL_TIME_DURATION = "Total_Time_Duration"
    WELL = "Well"
