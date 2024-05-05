from autogluon.tabular import TabularPredictor
import pandas as pd
import os



class AutoGluonTabular:

    def __init__(self, problem_type, eval_metric, time_limit, excluded_model_types=[]):

        self.params_model = {
                        'problem_type': problem_type,
                        'eval_metric': eval_metric,
        }
        
        self.params_fit = {
                        'time_limit': time_limit,
                        'excluded_model_types': excluded_model_types,
                        'save_space': True,
                        'presets': 'optimize_for_deployment', # for faster convergence
        }
        
    
    def _fit_predict(self, target, dir_models, X_train, X_val=None):
        model = TabularPredictor(
                    label = target,
                    path = dir_models,
                    verbosity = 0,
                    **self.params_model,
        )

        model.fit(  train_data  = X_train,
                    tuning_data = X_val,
                    **self.params_fit,
        )
        
        model.save()

        preds = model.predict(X_val, as_pandas=False)
    
        return preds



    def score_cv(self, X_train, target, cv_idx, dir_models, transformer=None, postprocessing=None, df_original=None):
        
        df_oof = pd.DataFrame(index=X_train.index, columns=[target])

        for fold, (idx_train, idx_val) in enumerate(cv_idx):
            idx_train = list(set(idx_train) & set(X_train.index))
            idx_val = list(set(idx_val) & set(X_train.index))
            Xt = X_train.loc[idx_train]
            if df_original is not None:
                Xt = pd.concat([Xt, df_original], axis=0)
            Xv = X_train.loc[idx_val]

            if transformer is not None:
                Xt = transformer.fit_transform(Xt)
                Xv = transformer.transform(Xv)

            preds = self._fit_predict(target, 
                                      dir_models + f'_fold_{fold}', 
                                      Xt, Xv)
            if postprocessing is not None:
                preds = postprocessing.transform(preds)
            df_oof.loc[idx_val, target] = preds

        return df_oof
    

    def predict(self, dir_model, df_input, transformer=None, postprocessing=None):
        model = TabularPredictor.load(dir_model)

        df = df_input
        if transformer is not None:
            df = transformer.transform(df.copy())

        preds = model.predict(df, as_pandas=False)
        if postprocessing is not None:
            preds = postprocessing.transform(preds)

        return preds


    def predict_cv(self, dir_models, df_input, transformer=None, postprocessing=None):
        preds = []
        for fold, path in enumerate(os.listdir(dir_models)):
            p = self.predict(os.path.join(dir_models, path), df_input, transformer, postprocessing)
            preds.append(p)

        return preds