"""
This program is a module for processing the data specified.
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

class Preprocessing(pd.DataFrame):
    """
    This represents the class for processing the contents of a specified data in preparation for applying data analytics techniques.
    """
    version = "1.0"
    def __init__(self, df_input=None):  
        """

        Initializes the use of the class and its functions 
        
        If a dataframe input is specified, it initializes that dataframe as the source of data that will be used throughout the program

        Parameters
        ----------
        df_input : str, optional
            the data frame where the visualizations will be based from
        """

        if df_input is None:
            pass
        else:
            try:
                self.df_input = pd.read_csv(df_input)
                self.target = None

            except Exception as e:
                print(e)

    def get_df(self):
        """
        Returns the initialized dataframe

        Returns
        -------
        pandas Dataframe
            initialized dataframe
        """

        try:
            return self.df_input
        except Exception as e:
            print(e)

    def set_new_df(self, new_df):
        """

        Sets a new dataframe

        Parameters
        ----------
        new_df : str, pandas Dataframe
            the dataframe where the visualizations will be based from
        """

        try:
            if(isinstance(new_df, pd.DataFrame)):
                self.df_input = new_df
            if(isinstance(new_df, str)):
                self.df_input = pd.read_csv(new_df)
        except Exception as e:
            print(e)

    def get_column(self, from_int=None, to_int=None):
        """

        Returns the columns of the dataframe

        If a specific range is given, it will return the column names of the dataframe within that specific range

        Parameters
        ----------
        from_int : int, optional
            the index start of the column names to be returned

        to_int : int, optional
            the index end of the column names to be returned

        Returns
        -------
        list
            list of column names
        """

        try:
            if from_int is None and to_int is None: 
                return list(self.df_input)
            else:
                get_col_arr = list(self.df_input)
                column_arr = []
                while from_int < to_int:
                    column_arr.append(get_col_arr[from_int])
                    from_int += 1
                return column_arr
        except Exception as e:
            print(e)

    def print_column(self, from_int=None, to_int=None):
        """

        Prints the columns of the dataframe

        If a specific range is given, it will print the column names of the dataframe within that specific range

        Parameters
        ----------
        from_int : int, optional
            the index start of the column names to be printed
        to_int : int, optional
            the index end of the column names to be printed
        """

        try:
            if from_int is None and to_int is None:
                counter = 1
                for col in list(self.df_input): 
                    print(str(counter) + " " + col) 
                    counter += 1
            else:
                counter = 1
                print_col_arr = list(self.df_input)
                while from_int < to_int:
                    print(str(counter) + " " + print_col_arr[from_int])
                    from_int += 1
                    counter += 1
        except Exception as e:
            print(e)

    def print_arr(self, inp_df):
        """

        Prints the contents of the given list with counters from 1 to end of the length of the list

        Parameters
        ----------
        inp_df : list
            the list to be printed with counters
        """

        try:
            counter = 0
            while counter < len(inp_df):
                print(str(counter+1) + " " + inp_df[counter])
                counter += 1
        except Exception as e:
            print(e)

    def get_row(self, identifier, from_int=None, to_int=None):
        """

        Returns the rows of a specified column in the dataframe

        Returns a list that contains the rows of the dataframe within a specific range and identifier

        Parameters
        ----------
        identifier : str
            name of the columns
        from_int : int, optional
            the index start of the row contents to be returned
        to_int : int, optional
            the index end of the row contents to be returned

        Returns
        -------
        pandas Dataframe
            rows of the specified column

        list
            alternatively returns rows of the specified column and within the specified range
        """
        
        try:
            if from_int is None and to_int is None: 
                return self.df_input[identifier]
            else:
                get_row_arr = self.df_input[identifier]
                row_arr = []
                while from_int < to_int:
                    row_arr.append(get_row_arr[from_int])
                    from_int += 1
                return row_arr
        except Exception as e:
            print(e)

    def print_row(self, identifier, from_int=None, to_int=None):
        """

        Prints the rows of a specified column in the dataframe

        Prints all the rows of the dataframe within a specific range and identifier 

        Parameters
        ----------
        identifier : str
            name of the columns
        from_int : int, optional
            the index start of the row contents to be returned
        to_int : int, optional
            the index end of the row contents to be returned
        """

        try:
            if from_int is None and to_int is None:
                counter = 1
                for col in self.df_input[identifier]: 
                    print(str(counter) + " " + col) 
                    counter += 1
            else:
                counter = 1
                arr = self.df_input[identifier]
                while from_int < to_int:
                    print(str(counter) + " " + arr[from_int])
                    from_int += 1
                    counter += 1
        except Exception as e:
            print(e)

    def locate(self, column, cond_inp):
        """

        Returns a boolean Series based on locating certain rows of a specified column which satisfies a specified condition
        
        Parameters
        ----------
        column : str
            the column of the dataframe where the rows will located at and compared with cond_inp
        cond_inp : str
            the conditional input that will be compared with the contents of the rows of the specified column

        Returns
        -------
        boolean Series
            series containing rows of a specified column which satisfies a specified condition
        """

        try:
            return self.df_input.loc[self.df_input[column] == cond_inp]
        except Exception as e:
            print(e)
  
    def group_frame_from(self, identifier, from_int, to_int, df_input=None):
        """

        Returns a Series which is grouped based on a certain identifier and a specific column range

        Parameters
        ----------
        identifier : str
            the identifier which will serve as the basis for grouping the dataframe
        from_int : int
            the index start of the column/s to be grouped
        to_int: int
            the index end of the column/s to be grouped
        df_input: pandas Dataframe, optional
            custom dataframe to be grouped

        Returns
        -------
        Series
            series containing the grouped rows based on a certain identifier and a specific column range
        """

        try:
            if df_input is None:
                first_df = self.df_input.groupby([identifier],as_index=False)[self.df_input.columns[from_int:to_int]].sum()
                return first_df
            else:
                first_df = df_input.groupby([identifier],as_index=False)[df_input.columns[from_int:to_int]].sum()
                return first_df
        except Exception as e:
            print(e)
      
    def group_frame_except(self, identifier, from_int, to_int, df_input=None):
        """

        Returns a Series which is grouped based on a certain identifier and a specific column range wherein the outliers of the specified index end are excluded from grouping but will still be part of the Series returned

        Parameters
        ----------
        identifier : str
            the identifier which will serve as the basis for grouping the dataframe
        from_int : int
            the index start of the column/s to be grouped
        to_int : int
            the index end of the column/s to be grouped
        df_input : pandas Dataframe, optional
            custom dataframe to be grouped
        
        Returns
        -------
        Series
            series containing the grouped rows based on a certain identifier and a specific column range with the exception of the outliers
        """

        try:
            if df_input is None:
                first_df = self.df_input.groupby([identifier],as_index=False)[self.df_input.columns[from_int:to_int]].sum()
                second_df = self.df_input.iloc[: , to_int:]
                return first_df.join(second_df) 
            else:
                first_df = df_input.groupby([identifier])[df_input.columns[from_int:to_int]].sum()
                second_df = df_input.iloc[: , to_int:]
                return first_df.join(second_df) 
        except Exception as e:
            print(e)

    def extract_row(self, column, identifier, df_input=None):
        """

        Returns a boolean Series containing the content of rows based on the specified column which matches a specific identifier
        
        Parameters
        ----------
        column : str
            the column of the dataframe where the rows will extracted at
        identifier : str
            the identifier of the rows to be extracted
        df_input : pandas Dataframe, optional
            custom dataframe to be grouped

        Returns
        -------
        boolean Series
            series containing rows of a specific column which matches a specific identifier
        """

        try:
            if df_input is None:
                return self.df_input.loc[self.df_input[column] == identifier]
            else:
                return df_input.loc[df_input[column] == identifier]  
        except Exception as e:
            print(e)

    def scaler(self, columns=None, minmax=(0,1)):
        scaler = MinMaxScaler(feature_range= minmax)
        if(self.target==None):
            self.df = scaler.fit_transform(self.df.astype(float))
        else:
            for col in self.df_input.columns:
                if(col !=self.target):
                    self.df[col] = scaler.fit_transform(self.df[[col]].astype(float))

    def bin(self,n,column, strat="uniform"):
        valid_strategy = ('uniform', 'quantile', 'kmeans')
        if strat not in valid_strategy:
            raise ValueError("Valid options for 'bin_strat' are {}. "
                             "Got strategy={!r} instead."
                             .format(valid_strategy, strat))
        est = KBinsDiscretizer(n_bins=n, encode='ordinal', strategy=strat)
        self.df_input[column] = est.fit_transform(self.df_input[column])
        print(self.df_input[column])

    def feature_select(self, X, y, cv_kfold=5, min_features=None):
        try:
            min_features = 1 if min_features==None else min_features

            selector = RFECV(LogisticRegression(), step=1,cv=cv_kfold, min_features_to_select=min_features)
            selector = selector.fit(X,y)
            support = selector.support_
            selected = []
            for a, s in zip(X.columns, support):
                if(s):
                    selected.append(a)

            return pd.concat([X[selected],y], axis=1)

        except Exception as e:
            print(e)