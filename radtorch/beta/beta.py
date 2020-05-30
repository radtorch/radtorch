def balance_dataframe(dataframe, label_col, method='upsample'):
    df = dataframe
    counts=df.groupby(label_col).count()
    classes=df[label_col].unique().tolist()
    max_class_num=counts.max()[0]
    max_class_id=counts.idxmax()[0]
    min_class_num=counts.min()[0]
    min_class_id=counts.idxmin()[0]
    if method=='upsample':
        resampled_subsets = [df[df[label_col]==max_class_id]]
        for i in [x for x in classes if x != max_class_id]:
          class_subset=df[df[label_col]==i]
          upsampled_subset=resample(class_subset, n_samples=max_class_num, random_state=100)
          resampled_subsets.append(upsampled_subset)
    elif method=='downsample':
        resampled_subsets = [df[df[label_col]==min_class_id]]
        for i in [x for x in classes if x != min_class_id]:
          class_subset=df[df[label_col]==i]
          upsampled_subset=resample(class_subset, n_samples=min_class_num, random_state=100)
          resampled_subsets.append(upsampled_subset)
    resampled_df = pd.concat(resampled_subsets)
    return resampled_df
