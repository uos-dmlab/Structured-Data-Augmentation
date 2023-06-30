import importlib

def call_dataset(dataset):
  global record_clustering
  global data_generation
  if dataset == 'tox21':
    data_generation = importlib.import_module("data_generation_tox21")
  else:
    data_generation = importlib.import_module("data_generation")

  if dataset=='diabetes':
    categorical_columns = ['cluster','Outcome','Pregnancies']
    mixed_columns = ['BMI','DiabetesPedigreeFunction']
    integer_columns = ['Glucose','BloodPressure','SkinThickness','Insulin','Age']
    label = 'Outcome'
  elif dataset in ['adult1000', 'adult5000', 'adult10000', 'adult20000']:
    categorical_columns = ['cluster','workclass', 'education', 'marital-status', 'occupation',
                                        'relationship', 'race', 'gender', 'native-country', 'income']
    mixed_columns = ['apital-loss','capital-gain']
    integer_columns = ['age', 'fnlwgt', 'capital-loss','hours-per-week']
    label = 'income'
  elif dataset=='online':
    categorical_columns = ['cluster','Revenue','Administrative','Informational','SpecialDay','Month','OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend']
    mixed_columns = ['Administrative_Duration','Informational_Duration','ProductRelated_Duration','BounceRates','ExitRates','PageValues']
    integer_columns = ['ProductRelated']
    label = 'Revenue'
  elif dataset=='churn':
    categorical_columns = ['cluster','Churn','DeviceProtection','PaperlessBilling','PaymentMethod','gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','TechSupport','StreamingTV','StreamingMovies']
    mixed_columns = ['MonthlyCharges','TotalCharges']
    integer_columns = ['tenure','Contract']
    label = 'Churn'
  elif dataset=='car':
    categorical_columns = ['cluster','OUTCOME','GENDER','RACE','DRIVING_EXPERIENCE','EDUCATION','INCOME','VEHICLE_OWNERSHIP','VEHICLE_YEAR','MARRIED','POSTAL_CODE','VEHICLE_TYPE','DUIS']
    mixed_columns = ['CREDIT_SCORE']
    integer_columns = ['AGE','ANNUAL_MILEAGE','CHILDREN','SPEEDING_VIOLATIONS','PAST_ACCIDENTS']
    label = 'OUTCOME'
  elif dataset=='bank':
    categorical_columns = ['cluster','job','marital','education','default','housing','loan','contact','month','campaign','previous','poutcome','Target']
    mixed_columns = ['balance','duration']
    integer_columns = ['age','day','pdays']
    label = 'Target'
  elif dataset=='credit_screening':
    categorical_columns = ['cluster','class','col_1','col_4','col_5','col_6','col_7','col_9','col_10','col_11','col_12','col_13']
    mixed_columns = ['CREDIT_SCORE']
    integer_columns = ['col_2','col_3','col_8','col_14','col_15']
    label = 'class'
  elif dataset=='tox21':
    categorical_columns = ['289', '208', '397', '304', '309','SR.ARE','cluster']
    mixed_columns = ['696', '426', '569', '644', '79']
    integer_columns = ['115', '27', '97', '465', '7', '773', '564', '488', '32', '477', '278', '219', '210', '261', '173', '772', '98', '126', '187', '135', '249', '680', '81', '602', '675', '205', '561', '372', '718', '452', '399', '422', '138', '753', '795', '226', '282', '134', '148', '497', '570', '586', '378', '228', '345', '432', '486', '240', '424', '505', '474', '376', '779', '387', '615', '327', '20', '591', '82', '584', '596', '175', '634', '375', '447', '265', '171', '518', '589', '655', '545', '45', '467', '635', '562', '566', '750', '71', '230', '420', '405', '402', '419', '350', '199', '515', '534', '380', '471', '503']
    label = 'SR.ARE'


  df_raw = './data/'+str(dataset)+'.csv'
  record_clustering_path = './record_clustering/'+str(dataset)+'_record_clustering.csv'
  gan_path = './GAN/'+str(dataset)+'_gan.csv'

  return {
            "categorical_columns": categorical_columns,
            "mixed_columns": mixed_columns,
            "integer_columns": integer_columns,
            "label": label,
            "df_raw": df_raw,
            "record_clustering_path": record_clustering_path,
            "gan_path": gan_path,
            "name":dataset 
      }
