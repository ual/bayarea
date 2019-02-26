# -*- coding: utf-8 -*-

import pandas as pd
import tabulate
import yaml
import os


class model:
    def __init__(self, filename):
        self.filename = filename
        
        with open("../lcog/configs/"+filename+".yaml", 'r') as stream:
            self.yaml_content = yaml.load(stream)
        
        self.model_type = self.yaml_content['saved_object']['template']
        self.title = "**{} Model**".format(filename.upper())
        
        sumlines = self.yaml_content['saved_object']['summary_table'].splitlines()
        separators = [i for i, line in enumerate(sumlines) if line.startswith(('=','-'))]
        
        scores = []
        [[scores.extend(i.strip().split('   ')) for i in line.split(':')] for line in sumlines[separators[0] + 1:separators[1]]]

        rows = [i.split() for i in sumlines[separators[2] + 1:separators[3]]]
        
        if self.model_type == 'LargeMultinomialLogitStep':
            self.segment = "* Agent segment: {}".format(self.yaml_content['saved_object']['out_chooser_filters'][-1])
            
            a = [i for i, sc in enumerate(scores) if sc == 'Log-Likelihood']
            loglike = float(scores[a[0]+1].replace(',',''))
            a = [i for i, sc in enumerate(scores) if sc == 'LL-Null']
            llnull = float(scores[a[0]+1].replace(',',''))
            log_ratio = abs(loglike/llnull)
            
            a = [i for i, sc in enumerate(scores) if sc == 'Pseudo R-squ.']
            r_squared = scores[a[0]+1]         
            
            cols = ['Variable','Coefficient','Std. Error','Z-score', 'P>|z|']
            df = pd.DataFrame(columns= cols, data = rows )
            df.drop(columns=['P>|z|'],inplace=True)
            table_md = tabulate.tabulate(df, tablefmt="pipe", headers="keys")
            self.table_md = table_md
            self.descrip = '* Log likelihood ratio (measure of fit): {}\n* R²: {}\n'.format(
                            log_ratio , r_squared )
            
        elif self.model_type == 'OLSRegressionStep':
            self.segment = "* Agent segment: {}".format(self.yaml_content['saved_object']['filters'].split(') & (')[-1].strip(')'))
            
            cols = ['Variable','Coefficient','Std. Error','T-score', 'p>|t|','interval','interval2']
            df = pd.DataFrame(columns= cols, data = rows )
            df.drop(columns=['p>|t|','interval','interval2'],inplace=True)
            table_md = tabulate.tabulate(df, tablefmt="pipe", headers="keys")
            self.table_md = table_md
            a = [i for i, sc in enumerate(scores) if sc == 'F-statistic']
            f_statistic = scores[a[0]+1]
            a = [i for i, sc in enumerate(scores) if sc == 'R-squared']
            r_squared = scores[a[0]+1]
            
            self.descrip = '* F-statistic: {}\n* R²: {}'.format(
                            f_statistic , r_squared )
        
        elif self.model_type == 'BinaryLogitStep':
            self.segment = "* Agent segment: {}".format(self.yaml_content['saved_object']['tables'])

            a = [i for i, sc in enumerate(scores) if sc == 'Log-Likelihood']
            loglike = float(scores[a[0]+1].replace(',',''))
            a = [i for i, sc in enumerate(scores) if sc == 'LL-Null']
            llnull = float(scores[a[0]+1].replace(',',''))
            log_ratio = abs(loglike/llnull)
            
            a = [i for i, sc in enumerate(scores) if sc == 'Pseudo R-squ.']
            r_squared = scores[a[0]+1] 

            cols = ['Variable','Coefficient','Std. Error','Z-score', 'P>|z|', 'interval', 'interval2' ]
            df = pd.DataFrame(columns= cols, data = rows )
            df.drop(columns=['P>|z|', 'interval', 'interval2'],inplace=True)
            table_md = tabulate.tabulate(df, tablefmt="pipe", headers="keys")
            self.table_md = table_md
            
            self.descrip = '* Log likelihood ratio (measure of fit): {}\n* R²: {}\n'.format(
                            log_ratio , r_squared )
            
        self.tabletitle = 'Estimated coefficients for {}:'.format(filename)

    def summary_to_md_file(self):
        all_text = '\n\n{}\n\n{}\n{}\n{}\n\n{}\n'.format(self.title, self.segment, self.descrip, self.tabletitle, self.table_md)
        
        file_name="Model-Estimation-Results"
        with open('{}.md'.format(file_name), mode='a') as md_file:
            md_file.write('\n' + all_text)

#Start with Global description
file_name="Model-Estimation-Results"
intro_text = "This page documents choice model estimation results for the parcel-level LCOG UrbanSim model." \
    +"The location choice models drive the placement of new and relocating agents in the simulation." \
    +"The Relocation choice models estimates de moving of households from their current location in the simulation." \
    +"The model is estimated previously to the location choice models since this instance will determine the relocation or not and therefore will create the need of finding another building to moving households." \
    +"The Tenure choice models models households' decisions to own or rent the place they live. This model is applied after estimating the new location with the hlcm." \
    +"The price regression models estimate the variation of buildings' values.\n"\
+"For each submodel, a table of estimated coefficients/significances are presented. "\
+"Measures of fit and other model evaluation metrics accompany the coefficient tables.\n\n"\
+"Model acronyms are defined as:\n"\
+"HLCM: Household location choice model\n"\
+"household_relocation_choice_model: Household Relocation choice model\n"\
+"tenure_choice_model: Household Tenure choice model\n"\
+"ELCM: Employment location choice model\n"\
"REPM: Residential/Employment Price model\n"
# NRDPLCM: Non-residential development project location choice model
# RDPLCM: Residential development project location choice model'
with open('{}.md'.format(file_name), mode='w') as md_file:
    md_file.write(intro_text)

# Print every model type with the corresponding big title
models_types_dict = {'hlcm': 'Household Location Choice Models' ,
                    'household_relocation_choice_model': 'Household Relocation Choice Model' ,
                    'tenure_choice_model': 'Household Tenure Choice Model' ,
                     'elcm': 'Employment Location Choice Models',
                     'repm': 'Residential/Employment Price Models'}
for typ in models_types_dict.keys():
    title = models_types_dict[typ]
    text_title = '\n## {}'.format(title)
    with open('{}.md'.format(file_name), mode='a') as md_file:
        md_file.write(text_title)

    files_typ= [name.split('.')[0] for name in os.listdir("../lcog/configs/") if name.startswith(typ)]
    for each_model in files_typ:
        print(each_model)
        model_class = model(each_model)
        model_class.summary_to_md_file()
