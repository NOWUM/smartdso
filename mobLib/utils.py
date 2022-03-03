import pandas as pd


def read_MIT_data(path: str = r'./mobLib/data/'):

    work = dict(
        days=pd.read_excel(fr'{path}MIT_Data_Work.xlsx', sheet_name='Days', index_col=0),
        times=pd.read_excel(fr'{path}MIT_Data_Work.xlsx', sheet_name='Times', index_col=0),
        distances=pd.read_excel(fr'{path}MIT_Data_Work.xlsx', sheet_name='Distances', index_col=0),
        durations=pd.read_excel(fr'{path}MIT_Data_Work.xlsx', sheet_name='Durations', index_col=0),
        usage=pd.read_excel(fr'{path}MIT_Data_Work.xlsx', sheet_name='Usage', index_col=0),
        type=pd.read_excel(fr'{path}MIT_Data_Work.xlsx', sheet_name='Type', index_col=0)
    )

    errand = dict(
        days=pd.read_excel(fr'{path}MIT_Data_Errand.xlsx', sheet_name='Days', index_col=0),
        times=pd.read_excel(fr'{path}MIT_Data_Errand.xlsx', sheet_name='Times', index_col=0),
        distances=pd.read_excel(fr'{path}MIT_Data_Errand.xlsx', sheet_name='Distances', index_col=0),
        durations=pd.read_excel(fr'{path}MIT_Data_Errand.xlsx', sheet_name='Durations', index_col=0),
        usage=pd.read_excel(fr'{path}MIT_Data_Errand.xlsx', sheet_name='Usage', index_col=0),
        freq=pd.read_excel(fr'{path}MIT_Data_Errand.xlsx', sheet_name='Freq', index_col=0)
    )

    hobby = dict(
        days=pd.read_excel(fr'{path}MIT_Data_Hobby.xlsx', sheet_name='Days', index_col=0),
        times=pd.read_excel(fr'{path}MIT_Data_Hobby.xlsx', sheet_name='Times', index_col=0),
        distances=pd.read_excel(fr'{path}MIT_Data_Hobby.xlsx', sheet_name='Distances', index_col=0),
        durations=pd.read_excel(fr'{path}MIT_Data_Hobby.xlsx', sheet_name='Durations', index_col=0),
        usage=pd.read_excel(fr'{path}MIT_Data_Hobby.xlsx', sheet_name='Usage', index_col=0)
    )

    return work, errand, hobby


if __name__ == "__main__":

    x = read_MIT_data()