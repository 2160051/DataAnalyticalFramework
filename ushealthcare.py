import os
import warnings
import platform
import pandas as pd
import numpy as np
from framework import DataAnalyticalFramework
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

#file_path = "Data Analytics.csv"
file_path = os.path.dirname(__file__) + "/Data Analytics.csv"
test_frame = DataAnalyticalFramework(file_path)


def clear_console():
    if(platform.system() == "Windows"):
        os.system("cls")
    elif(platform.system() == "Linux" or platform.system() == "Darwin"):
        os.system("clear")

def input_validation():
    inp = 0
    while True:
        try:
            inp = int(input())            
            clear_console()
            break
        except ValueError:
            print("Input must be an integer. Try again")
    return inp

def km_nb_df(test_frame):
    disease_arr = test_frame.get_column(3, 54)
    attribute_arr = test_frame.get_column(54,80)
    clear_console()
    return disease_arr, attribute_arr

def regression_df(test_frame):
    test_frame.set_new_df(test_frame.group_frame_except("State", 3, 49))
    disease_arr = test_frame.get_column(1, 47)
    attribute_arr = test_frame.get_column(52, 78)
    clear_console()
    return disease_arr, attribute_arr

def regression_df_state(test_frame):
    state_arr = test_frame.get_row("State", 0, 50)
    disease_arr = test_frame.get_column(3, 49)
    attribute_arr = test_frame.get_column(54, 80)
    clear_console()
    return state_arr, disease_arr,attribute_arr

def regression_df_week(test_frame):
    disease_arr = test_frame.get_column(3, 49)
    attribute_arr = test_frame.get_column(54, 80)
    clear_console()
    return disease_arr, attribute_arr

def regression_df_categories_week(test_frame):
    disease_arr = test_frame.get_column(49, 54)
    attribute_arr = test_frame.get_column(54, 80)
    clear_console()
    return disease_arr, attribute_arr

def regression_df_categories_state(test_frame):
    test_frame.set_new_df(test_frame.group_frame_except("State", 3, 49))
    disease_arr = test_frame.get_column(47, 52)
    attribute_arr = test_frame.get_column(52, 78)
    clear_console()
    return disease_arr, attribute_arr

def regression_df_categories(test_frame):
    state_arr = test_frame.get_row("State", 0, 50)
    disease_arr = test_frame.get_column(49, 54)
    attribute_arr = test_frame.get_column(54, 80)
    clear_console()
    return state_arr, disease_arr, attribute_arr


def week_list():
    while True:
        print("----------Select Week-----------")
        print("Select a week from week 1 - 26")
        print("27 Back to Start")
        print("----------Input-----------")
        print("Enter the week (#) of your choice: ")
        week_inp = input_validation()
        clear_console()
        if(week_inp == 27):
            main()
        elif(week_inp>27):
            print('Invalid Input')
        else:
            break
    return week_inp

def state_list(state_arr):
    while True:
        print("----------Select State-----------")
        test_frame.print_arr(state_arr)
        print(str(len(state_arr) + 1) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        state_inp = input_validation() - 1
        clear_console()
        if(state_inp == len(state_arr)):
            main()
        elif(state_inp>len(state_arr)):
            print('Invalid Input')
        else:
            break

    return state_inp

def attribute_list(attribute_arr):
    while True:
        print("----------Select Attribute-----------")
        test_frame.print_arr(attribute_arr)
        print(str(len(attribute_arr) + 1) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        attribute_inp = input_validation() - 1
        clear_console()
        
        if(attribute_inp == len(attribute_arr)):
            main()
        elif(attribute_inp>len(attribute_arr)):
            print('Invalid Input')
        else:
            break
    return attribute_inp

def disease_list(disease_arr):
    while True:
        print("----------Select Notifiable Disease-----------")
        test_frame.print_arr(disease_arr)
        print(str(len(disease_arr) + 1) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        disease_inp = input_validation() - 1
        clear_console()

        if(disease_inp == len(disease_arr)):
            main()
        elif(disease_inp>len(disease_arr)):
            print('Invalid Input')
        else:
            break
    return disease_inp

def disease_category_list(disease_arr):
    while True:
        print("----------Select Category of Disease to Compare-----------")
        test_frame.print_arr(disease_arr)
        print(str(len(disease_arr) + 1) + " All Categories")
        print(str(len(disease_arr) + 2) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        disease_inp = input_validation()
        clear_console()

        if(disease_inp == len(disease_arr) + 2):
            main()
        elif(disease_inp>len(disease_arr)+3):
            print('Invalid Input')
        else:
            break
    return disease_inp

def save_csv(figure):
    while True:
        if(figure == 13):
            print("----------Would You Like to Save the Clustered Dataset in a CSV File?-----------")
        else:
            print("----------Would You Like to Save it in a CSV File?-----------")
        print("1 Yes")
        print("2 No. Go Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        save_inp = input_validation()
        clear_console()

        if(save_inp==0 or save_inp>2):
            print("Invalid Input")
        elif(save_inp == 2):
            main()
        else:
            break
    return save_inp

def regression_options(choice,figure):
    if(choice == 1):
        disease_arr, attribute_arr= regression_df(test_frame)
        disease_inp = disease_list(disease_arr)
        attribute_inp = attribute_list(attribute_arr)
        if(figure == 1):
            test_frame.scatter_plot(attribute_arr[attribute_inp], disease_arr[disease_inp])
    elif(choice == 2):
        state_arr, disease_arr, attribute_arr = regression_df_state(test_frame)
        state_inp = state_list(state_arr)
        disease_inp = disease_list(disease_arr)
        attribute_inp = attribute_list(attribute_arr)
        test_frame.set_new_df(test_frame.extract_row("State", state_arr[state_inp]))
        if(figure == 1):
            test_frame.scatter_plot(attribute_arr[attribute_inp], disease_arr[disease_inp])
    elif(choice == 3):
        disease_arr, attribute_arr = regression_df_week(test_frame)
        week_inp = week_list()

        disease_inp = disease_list(disease_arr)
        attribute_inp = attribute_list(attribute_arr)
        test_frame.set_new_df(test_frame.locate("Week", week_inp))
        if(figure == 1):
            test_frame.scatter_plot(attribute_arr[attribute_inp], disease_arr[disease_inp])
    elif(choice == 4):
        disease_arr, attribute_arr = regression_df_categories_week(test_frame)
        week_inp = week_list()
        disease_inp = disease_list(disease_arr)
        attribute_inp = attribute_list(attribute_arr)
        test_frame.set_new_df(test_frame.locate("Week", week_inp))   
        if(figure == 1):
            test_frame.scatter_plot(attribute_arr[attribute_inp], disease_arr[disease_inp])
    elif(choice == 5):
        disease_arr, attribute_arr = regression_df_categories_state(test_frame)
        disease_inp = disease_list(disease_arr)
        attribute_inp = attribute_list(attribute_arr)
        if(figure == 1):
            test_frame.scatter_plot(attribute_arr[attribute_inp], disease_arr[disease_inp])
    elif(choice == 6):
        state_arr, disease_arr, attribute_arr = regression_df_categories(test_frame)
        state_inp = state_list(state_arr)
        disease_inp = disease_list(disease_arr)
        attribute_inp = attribute_list(attribute_arr)
        test_frame.set_new_df(test_frame.extract_row("State", state_arr[state_inp]))
        if(figure == 1):
            test_frame.scatter_plot(attribute_arr[attribute_inp], disease_arr[disease_inp])

    elif(choice == 7):
        main()

    if(figure==2 or figure == 5 ):
        while True:
            new_attrib_arr = attribute_arr.copy()
            new_attrib_arr.pop(attribute_inp)
            print("----------Select Another Attribute-----------")
            test_frame.print_arr(new_attrib_arr)
            print(str(len(new_attrib_arr) + 1) + " Back to Start")
            print("----------Input-----------")
            print("Enter the number (#) of your choice: ")
            attribute_second_inp = input_validation() - 1
            clear_console()
            if(attribute_second_inp == len(new_attrib_arr)):
                main()
            elif(attribute_second_inp>len(new_attrib_arr)):
                print("Invalid Input")
            else:
                break
        if (figure==2):
            test_frame.scatter_plot(attribute_arr[attribute_inp], disease_arr[disease_inp], new_attrib_arr[attribute_second_inp])
        elif(figure == 5):
            test_frame.linear_reg_summary(attribute_arr[attribute_inp], disease_arr[disease_inp],  new_attrib_arr[attribute_second_inp])
            print("Press enter to go back to main menu")
            input()
    
    elif(figure==3):
        test_frame.linear_regression(attribute_arr[attribute_inp], disease_arr[disease_inp])
    elif(figure==4):
        test_frame.linear_reg_summary(attribute_arr[attribute_inp], disease_arr[disease_inp])
        print("Press enter to go back to main menu")
        input()
    elif(figure==8):
        test_frame.polynomial_reg(attribute_arr[attribute_inp], disease_arr[disease_inp])
    elif(figure==9):
        test_frame.polynomial_reg_summary(attribute_arr[attribute_inp], disease_arr[disease_inp])
        print("Press enter to go back to main menu")
        input()


def table_summary(figure):
    
        test_frame.set_new_df(pd.read_csv(file_path))
        state_arr, disease_arr, attribute_arr = regression_df_categories(test_frame)
        state_inp = state_list(state_arr)

        test_frame.set_new_df(test_frame.extract_row("State", state_arr[state_inp]))
        attribute_inp = attribute_list(attribute_arr)
        
        if(figure == 7):
            new_attrib_arr = attribute_arr.copy()
            new_attrib_arr.pop(attribute_inp)

            attribute_second_inp = attribute_list(attribute_arr)


        person_to_person = ["Carbapenemase-producing carbapenem-resistant Enterobacteriaceae, Klebsiella spp.", "Chlamydia trachomatis infection", "Gonorrhea", "Hepatitis (viral, acute, by type), B", "Hepatitis (viral, acute, by type), C, Confirmed", "Hepatitis (viral, acute, by type), C, Probable", "Invasive Pneumococcal Disease, Age LT 5, Confirmed", "Invasive Pneumococcal Disease, Age LT 5, Probable", "Invasive Pneumococcal Disease, all ages, Confirmed","Invasive Pneumococcal Disease, all ages, Probable", "Meningococcal disease, all serogroups", "Shigellosis", "Syphilis, primary and secondary"]
        airborne = ["Coccidioidomycosis", "Legionellosis", "Mumps", "Pertussis"]
        droplet_spread = ["Haemophilus influenzae, invasive disease (all ages, all serotypes)", "Invasive Pneumococcal Disease, Age LT 5, Confirmed", "Invasive Pneumococcal Disease, Age LT 5, Probable", "Invasive Pneumococcal Disease, all ages, Confirmed", "Invasive Pneumococcal Disease, all ages, Probable", "Meningococcal disease, all serogroups", "Mumps", "Pertussis", "Rubella", "Rubella, congenital syndrome"]
        vector_borne = ["Babesiosis", "Dengue Virus Infections, Dengue", "Dengue Virus Infections, Severe Dengue", "Ehrlichiosis and Anaplasmosis, Anaplasma phagocytophilum infection", "Ehrlichiosis and Anaplasmosis, Ehrlichia chaffeensis infection", "Ehrlichiosis and Anaplasmosis, Ehrlichia ewingii infection", "Ehrlichiosis and Anaplasmosis, Undetermined Ehrlichiosis/Anaplasmosis", "Malaria", "Rabies, animal", "Spotted Fever Rickettsiosis, Confirmed", "Spotted Fever Rickettsiosis, Probable", "Tetanus", "Varicella morbidity", "West Nile virus disease, Neuroinvasive", "West Nile virus disease, Nonneuroinvasive", "Zika virus disease, non-congenital"]
        vehicle_borne = ["Campylobacteriosis", "Carbapenemase-producing carbapenem-resistant Enterobacteriaceae, Enterobacter spp.", "Carbapenemase-producing carbapenem-resistant Enterobacteriaceae, Escherichia coli", "Cryptosporidiosis", "Giardiasis", "Hepatitis (viral, acute, by type), A", "Hepatitis (viral, acute, by type), B", "Hepatitis (viral, acute, by type), C, Confirmed", "Hepatitis (viral, acute, by type), C, Probable", "Salmonellosis (excluding Paratyphoid fever andTyphoid fever)", "Shiga toxin-producing Escherichia coli", "Shigellosis", "Syphilis, primary and secondary", "Vibriosis (Any species of the family Vibrionaceae, other than toxigenic Vibrio cholerae O1 or O139), Confirmed", "Vibriosis (Any species of the family Vibrionaceae, other than toxigenic Vibrio cholerae O1 or O139), Probable"]
        categorized_disease = [person_to_person, droplet_spread, airborne, vector_borne, vehicle_borne]

        disease_inp = disease_category_list(disease_arr)

        if(disease_inp == 1):
            if(figure == 6):
                print(test_frame.linear_reg_table("Person-to-Person Transmission", attribute_arr[attribute_inp], categorized_disease[0]))
            elif(figure == 7):
                print(test_frame.linear_reg_table("Person-to-Person Transmission", attribute_arr[attribute_inp], categorized_disease[0], new_attrib_arr[attribute_second_inp]))

            save_inp = save_csv(figure)

            if(save_inp == 1):
                if(figure == 6):
                    table_df = test_frame.linear_reg_table("Person-to-Person Transmission", attribute_arr[attribute_inp], categorized_disease[0])
                    table_df.to_csv(os.path.dirname(__file__) + "/Person-to-Person Transmission and " + attribute_arr[attribute_inp] + ".csv")
                elif(figure == 7):
                    table_df = test_frame.linear_reg_table("Person-to-Person Transmission", attribute_arr[attribute_inp], categorized_disease[0], new_attrib_arr[attribute_second_inp])
                    table_df.to_csv(os.path.dirname(__file__) + "/Person-to-Person Transmission and " + attribute_arr[attribute_inp] + ".csv")
            elif(save_inp == 2):
                main()

        elif(disease_inp == 2):
            if(figure == 6):
                print(test_frame.linear_reg_table("Air-borne Transmission", attribute_arr[attribute_inp], categorized_disease[2]))
            elif(figure == 7):
                print(test_frame.linear_reg_table("Air-borne Transmission", attribute_arr[attribute_inp], categorized_disease[2], new_attrib_arr[attribute_second_inp]))
            
            save_inp = save_csv(figure)

            if(save_inp == 1):
                if(figure == 6):
                    table_df = test_frame.linear_reg_table("Air-borne Transmission", attribute_arr[attribute_inp], categorized_disease[2])
                    table_df.to_csv(os.path.dirname(__file__) + "/Air-borne Transmission and " + attribute_arr[attribute_inp] + ".csv")
                elif(figure == 7):
                    table_df = test_frame.linear_reg_table("Air-borne Transmission", attribute_arr[attribute_inp], categorized_disease[2], new_attrib_arr[attribute_second_inp])
                    table_df.to_csv(os.path.dirname(__file__) + "/Air-borne Transmission and " + attribute_arr[attribute_inp] + ".csv")
            elif(save_inp == 2):
                main()
        
        elif(disease_inp == 3):
            if(figure == 6):
                print(test_frame.linear_reg_table("Droplet Spread", attribute_arr[attribute_inp], categorized_disease[1]))
            elif(figure == 7):
                print(test_frame.linear_reg_table("Droplet Spread", attribute_arr[attribute_inp], categorized_disease[1], new_attrib_arr[attribute_second_inp]))
            
            save_inp = save_csv(figure)

            if(save_inp == 1):
                if(figure == 6):
                    table_df = test_frame.linear_reg_table("Droplet Spread", attribute_arr[attribute_inp], categorized_disease[1])
                    table_df.to_csv(os.path.dirname(__file__) + "/Droplet Spread and " + attribute_arr[attribute_inp] + ".csv")
                elif(figure == 7):
                    table_df = test_frame.linear_reg_table("Droplet Spread", attribute_arr[attribute_inp], categorized_disease[1], new_attrib_arr[attribute_second_inp])
                    table_df.to_csv(os.path.dirname(__file__) + "/Droplet Spread and " + attribute_arr[attribute_inp] + ".csv")
            elif(save_inp == 2):
                main()

        elif(disease_inp == 4):
            if(figure == 6):
                print(test_frame.linear_reg_table("Vector-borne Transmission", attribute_arr[attribute_inp], categorized_disease[3]))
            elif(figure == 7):
                print(test_frame.linear_reg_table("Vector-borne Transmission", attribute_arr[attribute_inp], categorized_disease[3], new_attrib_arr[attribute_second_inp]))
            
            save_inp = save_csv(figure)

            if(save_inp == 1):
                if(figure == 6):
                    table_df = test_frame.linear_reg_table("Vector-borne Transmission", attribute_arr[attribute_inp], categorized_disease[3])
                    table_df.to_csv(os.path.dirname(__file__) + "/Vector-borne Transmission and " + attribute_arr[attribute_inp] + ".csv")
                elif(figure == 7):
                    table_df = test_frame.linear_reg_table("Vector-borne Transmission", attribute_arr[attribute_inp], categorized_disease[3], new_attrib_arr[attribute_second_inp])
                    table_df.to_csv(os.path.dirname(__file__) + "/Vector-borne Transmission and " + attribute_arr[attribute_inp] + ".csv")

            elif(save_inp == 2):
                main()

        elif(disease_inp == 5):
            if(figure == 6):
                print(test_frame.linear_reg_table("Vehicle-borne Transmission", attribute_arr[attribute_inp], categorized_disease[4]))
            elif(figure == 7):
                print(test_frame.linear_reg_table("Vehicle-borne Transmission", attribute_arr[attribute_inp], categorized_disease[4], new_attrib_arr[attribute_second_inp]))
            
            save_inp = save_csv(figure)

            if(save_inp == 1):
                if(figure == 6):
                    table_df = test_frame.linear_reg_table("Vehicle-borne Transmission", attribute_arr[attribute_inp], categorized_disease[4])
                    table_df.to_csv(os.path.dirname(__file__) + "/Vehicle-borne Transmission and " + attribute_arr[attribute_inp] + ".csv")
                elif(figure == 7):
                    table_df = test_frame.linear_reg_table("Vehicle-borne Transmission", attribute_arr[attribute_inp], categorized_disease[4], new_attrib_arr[attribute_second_inp])
                    table_df.to_csv(os.path.dirname(__file__) + "/Vehicle-borne Transmission and " + attribute_arr[attribute_inp] + ".csv")
            elif(save_inp == 2):
                main()
                
        elif(disease_inp == 6):
            if(figure == 6):
                print(test_frame.linear_reg_table("Person-to-Person Transmission", attribute_arr[attribute_inp], categorized_disease[0]))
                print()
                print(test_frame.linear_reg_table("Air-borne Transmission", attribute_arr[attribute_inp], categorized_disease[2]))
                print()
                print(test_frame.linear_reg_table("Droplet Spread", attribute_arr[attribute_inp], categorized_disease[1]))
                print()
                print(test_frame.linear_reg_table("Vector-borne Transmission", attribute_arr[attribute_inp], categorized_disease[3]))
                print()
                print(test_frame.linear_reg_table("Vehicle-borne Transmission", attribute_arr[attribute_inp], categorized_disease[4]))
            elif(figure == 7):
                print(test_frame.linear_reg_table("Person-to-Person Transmission", attribute_arr[attribute_inp], categorized_disease[0], new_attrib_arr[attribute_second_inp]))
                print()
                print(test_frame.linear_reg_table("Air-borne Transmission", attribute_arr[attribute_inp], categorized_disease[2], new_attrib_arr[attribute_second_inp]))
                print()
                print(test_frame.linear_reg_table("Droplet Spread", attribute_arr[attribute_inp], categorized_disease[1], new_attrib_arr[attribute_second_inp]))
                print()
                print(test_frame.linear_reg_table("Vector-borne Transmission", attribute_arr[attribute_inp], categorized_disease[3], new_attrib_arr[attribute_second_inp]))
                print()
                print(test_frame.linear_reg_table("Vehicle-borne Transmission", attribute_arr[attribute_inp], categorized_disease[4], new_attrib_arr[attribute_second_inp]))
            

            save_inp = save_csv(figure)

            if(save_inp == 1):
                if(figure == 6):
                    table_df = test_frame.linear_reg_table("Person-to-Person Transmission", attribute_arr[attribute_inp], categorized_disease[0])
                    table_df.to_csv(os.path.dirname(__file__) + "/Person-to-Person Transmission and " + attribute_arr[attribute_inp] + ".csv")

                    table_df = test_frame.linear_reg_table("Droplet Spread", attribute_arr[attribute_inp], categorized_disease[1])
                    table_df.to_csv(os.path.dirname(__file__) + "/Droplet Spread and " + attribute_arr[attribute_inp] + ".csv")

                    table_df = test_frame.linear_reg_table("Air-borne Transmission", attribute_arr[attribute_inp], categorized_disease[2])
                    table_df.to_csv(os.path.dirname(__file__) + "/Air-borne Transmission and " + attribute_arr[attribute_inp] + ".csv")

                    table_df = test_frame.linear_reg_table("Vector-borne Transmission", attribute_arr[attribute_inp], categorized_disease[3])
                    table_df.to_csv(os.path.dirname(__file__) + "/Vector-borne Transmission and " + attribute_arr[attribute_inp] + ".csv")

                    table_df = test_frame.linear_reg_table("Vehicle-borne Transmission", attribute_arr[attribute_inp], categorized_disease[4])
                    table_df.to_csv(os.path.dirname(__file__) + "/Vehicle-borne Transmission and " + attribute_arr[attribute_inp] + ".csv")
                elif(figure == 7):
                    table_df = test_frame.linear_reg_table("Person-to-Person Transmission", attribute_arr[attribute_inp], categorized_disease[0], new_attrib_arr[attribute_second_inp])
                    table_df.to_csv(os.path.dirname(__file__) + "/Person-to-Person Transmission and " + attribute_arr[attribute_inp] + " and " + new_attrib_arr[attribute_second_inp] + ".csv")

                    table_df = test_frame.linear_reg_table("Droplet Spread", attribute_arr[attribute_inp], categorized_disease[1], new_attrib_arr[attribute_second_inp])
                    table_df.to_csv(os.path.dirname(__file__) + "/Droplet Spread and " + attribute_arr[attribute_inp] + " and " + new_attrib_arr[attribute_second_inp] + ".csv")

                    table_df = test_frame.linear_reg_table("Air-borne Transmission", attribute_arr[attribute_inp], categorized_disease[2], new_attrib_arr[attribute_second_inp])
                    table_df.to_csv(os.path.dirname(__file__) + "/Air-borne Transmission and " + attribute_arr[attribute_inp] + " and " + new_attrib_arr[attribute_second_inp] + ".csv")

                    table_df = test_frame.linear_reg_table("Vector-borne Transmission", attribute_arr[attribute_inp], categorized_disease[3], new_attrib_arr[attribute_second_inp])
                    table_df.to_csv(os.path.dirname(__file__) + "/Vector-borne Transmission and " + attribute_arr[attribute_inp] + " and " + new_attrib_arr[attribute_second_inp] + ".csv")

                    table_df = test_frame.linear_reg_table("Vehicle-borne Transmission", attribute_arr[attribute_inp], categorized_disease[4], new_attrib_arr[attribute_second_inp])
                    table_df.to_csv(os.path.dirname(__file__) + "/Vehicle-borne Transmission and " + attribute_arr[attribute_inp] + " and " + new_attrib_arr[attribute_second_inp] + ".csv")
                
            elif(save_inp == 2):
                main()
        elif(disease_inp == len(disease_arr) + 2):
            main()

def naive_bayes(figure):

        disease_arr, attribute_arr = km_nb_df(test_frame)
        while True:
            print("----------Select Notifiable Disease/Disease Category to use for the Naive-Bayes model -----------")
            test_frame.print_arr(disease_arr)
            print(str(len(disease_arr) + 1) + " Back to Start")
            print("----------Input-----------")
            print("Enter the number (#) of your choice: ")
            disease_inp = input_validation() - 1
            clear_console()
            if(disease_inp == len(disease_arr)):
                main()
            elif(disease_inp == len(disease_arr)+1 or disease_inp == 0):
                print("Invalid input")
            else:
                break
        while True:
            print("------ Binning ---------")
            print("#Split selected disease/disease category into n numbers")
            print("#Enter (0) if class/target is already categorized/binned")
            print("----------Input-----------")
            print("Enter number of Bins: ")
            n = input_validation()
            clear_console()
            if(n==1):
                print("Invalid Input")
            else:
                break

        while True:
            print("----------Binning Strategy-----------")
            print("1 Kmeans")
            print("2 Uniform ")
            print("----------Input-----------")
            print("Enter the number (#) of your choice: ")
            b = input_validation()   
            clear_console()
            if(b==1):
                s='kmeans'
                break
            elif(b==2):
                s='uniform'
                break
            else:
                print("Invalid Input")

        print("----------Feature Selection-----------")
        print("0 Skip Recursive Feature Selection (use all attributes).")
        print("1 Apply Recursive Feature Selection. ")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        fs = input_validation()
        if (fs==1):
            fs = True
        elif(fs==0):
            fs = False
        clear_console()

        while True:
            print("------ Cross Validation ---------")
            print("#Suggestions: 5, 10")
            print("#Enter (0) to skip cross validation")
            print("----------Input-----------")
            print("Enter the number of k to perform k-fold cross validation ")
            k = input_validation()
            clear_console()
            if(k==1):
                 print("Can't divide the data into "+str(k)+" folds.")
            else:
                break


        if(figure == 10):
            print(disease_arr[disease_inp])
            test_frame.naive_bayes_cm(attribute_arr, disease_arr[disease_inp], cv_kfold=k,class_bins=n, bin_strat=s, feature_selection = fs)
        elif(figure == 11):
            y_true, y_pred, accuracy = test_frame.naive_bayes(attribute_arr, disease_arr[disease_inp], cv_kfold=k, class_bins=n, bin_strat=s, feature_selection=fs)
            print("Accuracy: "+str(accuracy))
            print("Enter to continue")
            input()

def k_means(figure):

    disease_arr, attribute_arr = km_nb_df(test_frame)
    print("----------Select Notifiable Disease/Disease Category to use for k-means clustering -----------")
    test_frame.print_arr(disease_arr)
    print(str(len(disease_arr) + 1) + " Back to Start")
    print("----------Input-----------")
    print("Enter the number (#) of your choice: ")
    disease_inp = input_validation() - 1
    if(disease_inp == len(disease_arr)):
        main()
    elif(disease_inp>(len(disease_arr) + 1)):
        print("Invalid Input.")

    clear_console()
    print("----------K-means Clustering-----------")
    k = 0
    n = 0
    while(True):
        print("Enter the number of Clusters (k) that you want to generate:")
        k = input_validation()
        clear_console()
        if(k==0):
            print("Invalid input")
        else:
            break
    print("----------Feature Selection-----------")
    print("# Enter (0) to Use All Attributes.")
    while(True):
        print("----------Input-----------")
        print("Enter the number of features you want to use: ")
        n = input_validation()        
        clear_console()
        if(n>(len(attribute_arr)+1)):
            print("You entered"+str(k)+ "which is above the maximum number of features that can be used ("+ str(len(attribute_arr)+1)+")")
        else:
            break
    if(figure == 12):
        test_frame.k_means_cc(attribute_arr, disease_arr[disease_inp], k, n_features = n)
    elif(figure == 13):
        centroids, sil_coef,labeled_features = test_frame.k_means(attribute_arr, disease_arr[disease_inp], k, n_features =n)
        print("centroids: "+str(centroids))
        print("Silhouette Coefficient: "+str(sil_coef))
        save_inp = save_csv(figure)
        if(save_inp==1):
            labeled_features.to_csv(os.path.dirname(__file__) + "/" + disease_arr[disease_inp] + " clustered.csv")


def help_texts(help_inp):
    if(help_inp == 1):
        print("Scatter Plot - 2D data visualization which makes use of dots or points to represents the values of two different variables. It is sometimes referred to as correlation plot since it is used for showing the correlation or relationship between two variables.")
        print("Interpretation:")
        print("1. Positive Correlation - occurs when the values are increasing together. This is seen in a plot wherein the points are aligned to resemble a line leaning to the right.")
        print("2. Negative Correlation - occurs when the values are decreasing together. This is seen in a plot wherein the points are aligned to resemble a line leaning to the left.")
        print("3. No Correlation - occurs when the values are far from each other and has a shape that does not closely resembles a line.")
    elif(help_inp == 2):
        print("Scatter Plot with Multiple Attributes - 3D data visualization which makes use of dots or points to represents the values of three different variables. Instead of only making use of the x and y-axis, the z-axis is also used.")
        print("Interpretation:")
        print("1. Positive Correlation - occurs when the values are increasing together. This is seen in a plot wherein the points are aligned to resemble a line leaning to the right.")
        print("2. Negative Correlation - occurs when the values are decreasing together. This is seen in a plot wherein the points are aligned to resemble a line leaning to the left.")
        print("3. No Correlation - occurs when the values are far from each other and has a shape that does not closely resembles a line.")
    elif(help_inp == 3):
        print("Linear Regression - type of predictive analysis model for determining the relationship between two variables by fitting a line. One variable is the independent variable (x-axis) while the other is the dependent variable (y-axis). The analysis of a linear regression is based on its p-values and coefficients. This section covers only the visualization and the equation of the line of one independent variable and one dependent variable using linear regression. For calculating the result and hypothesis testing, please make use of the 5th choice in the type of figure which is 'Linear Regression, Results Only'.")
    elif(help_inp == 4):
        print("Multiple Linear Regression - type of predictive analysis model for determining the relationship between multiple variables by fitting a line. Unlike a simple linear regression, a multiple linear regression makes use of one dependent variable and two or more independent variable. For this program, only one dependent variable and two independent variables will be compared. The analysis is still based on the measured p-values and coefficients of the variables used.")
        print("Analysis:")
        print("1. P-value - of each independent variable tests the null hypothesis or the idea that the variable has no correlation with the dependent variable. It is used to determine the statistical significance of the correlation between the two variables. A low value, which is less than the indicated significance level or alpha (α), means that the null hypothesis can be rejected. On the other hand, a high p-value, more than the indicated significance level or alpha (α), means that the null hypothesis  can be accepted and that the changes in the independent variable does not affect the values of the dependent variable.")
        print("2. Coefficient - otherwise known as slope coefficient, it represents the mean change of the dependent variable as the independent variable changes. It is used for identifying the type of correlation between the variables, either positive or negative correlation. A positive correlation means that as the value of the independent variable increases, so does the mean of the dependent variable. The case is opposite in a negative correlation.")
    elif(help_inp == 5):
        print("Polynomial Curve - another way of analyzing regression wherein the relationship between the independent variable and dependent variable is modelled as a polynomial visualization of a certain specified degree. Unlike linear regression wherein the data visualization shows a line, polynomial regression takes the form of curves or other polynomial shapes.")
        print("Analysis:")
        print("1. P-value - of each independent variable tests the null hypothesis or the idea that the variable has no correlation with the dependent variable. It is used to determine the statistical significance of the correlation between the two variables. A low value, which is less than the indicated significance level or alpha (α), means that the null hypothesis can be rejected. On the other hand, a high p-value, more than the indicated significance level or alpha (α), means that the null hypothesis  can be accepted and that the changes in the independent variable does not affect the values of the dependent variable.")
        print("2. Coefficient - otherwise known as slope coefficient, it represents the mean change of the dependent variable as the independent variable changes. It is used for identifying the type of correlation between the variables, either positive or negative correlation. A positive correlation means that as the value of the independent variable increases, so does the mean of the dependent variable. The case is opposite in a negative correlation.")
    elif(help_inp == 6):
        print("Linear Regression - type of predictive analysis model for determining the relationship between two variables by fitting a line. One variable is the independent variable (x-axis) while the other is the dependent variable (y-axis). The analysis of a linear regression is based on its p-values and coefficients. This section covers only the visualization and the equation of the line of one independent variable and one dependent variable using linear regression. For displaying the visualization of the variables using simple linear regression, please make use of the 3rd choice in the type of figure which is 'Simple Linear Regression'.")
        print("Analysis:")
        print("1. P-value - of each independent variable tests the null hypothesis or the idea that the variable has no correlation with the dependent variable. It is used to determine the statistical significance of the correlation between the two variables. A low value, which is less than the indicated significance level or alpha (α), means that the null hypothesis can be rejected. On the other hand, a high p-value, more than the indicated significance level or alpha (α), means that the null hypothesis  can be accepted and that the changes in the independent variable does not affect the values of the dependent variable.")
        print("2. Coefficient - otherwise known as slope coefficient, it represents the mean change of the dependent variable as the independent variable changes. It is used for identifying the type of correlation between the variables, either positive or negative correlation. A positive correlation means that as the value of the independent variable increases, so does the mean of the dependent variable. The case is opposite in a negative correlation.")
    elif(help_inp == 7):
        print("Naive Bayes Classfication - a machine learning model that is used to predict a class based on specific features.")
        print("Analysis:")
        print("Confusion Matrix - shows the numbers of true and predicted values of the model on each bins.")
    elif(help_inp == 8):
        print("Naive Bayes Classfication - a machine learning model that is used to predict a class based on specific features.")
        print("Analysis:")
        print("accuracy - the balanced accuracy of the model.")
    elif(help_inp == 9):
        print("K-means Clustering - a method for clustering data into (k) number of clusters.")
        print("Analysis:")
        print("Set of Scatter Plots - a visualization of the clusters based on all the features used.")
        print("Silhouette Coefficient - a performance metric for checking the space between the clusters. The silhouette coefficient has a range of -1 to 1. A negative value indicates that data is may be the wrong cluster. A value close to 1 indicates that the clusters are at a distinctive space from each other. A value close to 0 indicates that the clusters are overlapping.")
        print("Centroid Chart - a line graph of the coordinates of the centroids.")
    elif(help_inp == 10):
        print("K-means Clustering - a method for clustering data into (k) number of clusters.")
        print("Analysis:")
        print("Table of the Clustered Dataset - a table with the rows of the dataset used in clustering with an additional column that indicates each row's cluster. The table can be saved as a CSV file.")
    elif(help_inp == 11):
        clear_console()
    elif(help_inp == 12):
        print("User has closed the program.")
        exit()


def regression_list1():
    while True:
        print("----------Would You Like To...-----------")
        print("1 Consider All States and Choose Attribute to Compare")
        print("2 Choose a State and Attribute to Compare")
        print("3 Choose by Week and Compare")
        print("4 Choose by Week and by Category of Disease and Compare")
        print("5 Choose Consider All States, by Category of Disease and Compare")
        print("6 Choose a State and by Category of Disease and Compare")
        print("7 Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        choice_inp = input_validation()
        clear_console()

        
        if(choice_inp>7):
            print('Invalid Input')
        else:
            break

    return choice_inp
        
def help_list():
        print("User sent a wrong input. Please try again.")
        print("----------Help-----------")
        print("1 Simple Scatter Plot")
        print("2 Scatter Plot with Multiple Attributes")
        print("3 Simple Linear Regression")
        print("4 Multiple Linear Regression")
        print("5 Polynomial Curve")
        print("6 Linear Regression, Results Only")
        print("7 Naive-Bayes Classification")
        print("8 Naive-Bayes Classification, Result Only")
        print("9 K-means")
        print("10 K-means, Result Only")
        print("11 Back to Start")
        print("12 Quit")
        print("----------Input-----------")
        print("Enter the number (#) of the chosen type of figure: ")
        help_inp = input_validation()
        return help_inp


def figure_list():
    while True:
        print("----------Select Type of Figure-----------")
        print("**Scatter Plot**")
        print("1 Simple Scatter Plot")
        print("2 Scatter Plot with Multiple Attributes")
        print()
        print("**Linear Regression**")
        print("3 Simple Linear Regression")
        print("4 Linear Regression, Results Only")
        print("5 Multiple Linear Regression")
        print("6 Table Summary Simple Linear Regression")
        print("7 Table Summary Multiple Linear Regression")
        print()
        print("**Polynomial Curve**")
        print("8 Polynomial Curve")
        print("9 Polynomial Curve, Results Only")
        print()
        print("**Naive-Bayes**")
        print("10 Naive-Bayes Classification")
        print("11 Naive-Bayes Classification, Result Only")
        print()
        print("**K-means Clustering**")
        print("12 K-means")
        print("13 K-means, Result Only")
        print()
        print("14 Help")
        print("15 Quit")
        print("----------Input-----------")
        print("Enter the number (#) of the chosen type of figure: ")
        fig_inp = input_validation()
        clear_console()

        if(fig_inp>15):
            print('Invalid Input')
        else:
            break
    return fig_inp

def main():
    figure = figure_list()
    if(figure <= 5 and figure > 0):
        choice = regression_list1()
        regression_options(choice,figure)
    elif(figure == 6 or figure == 7):
        table_summary(figure)

    elif(figure == 8 or figure == 9):
        choice = regression_list1()
        regression_options(choice,figure)
    
    elif(figure == 10 or figure == 11):
        naive_bayes(figure)
    
    elif(figure == 12 or figure == 13):
        k_means(figure)
    elif(figure == 14):
        while True:
            help_inp = help_list()
            help_texts(help_inp)
            print("---------------")
            print("1 back to help")
            inp = input_validation()
            clear_console()
            if(inp!=1):
                break
        
    elif(figure == 15):
        print("User has closed the program.")
        exit(0)
    test_frame.set_new_df(pd.read_csv(file_path))
    main()

main()