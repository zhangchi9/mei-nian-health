# Mei nian health

[![NPM Version][npm-image]][npm-url]
[![Build Status][travis-image]][travis-url]
[![Downloads Stats][npm-downloads]][npm-url]
[![License: MIT][mit-image]][mit-url]

![](header.jpg)

This is the Tianchi ￥250,000 house prices prediction challenge (https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100150.711.5.1ed42784KGi5ZD&raceId=231654). In the regression problem, I use different model to train the dataset, including Lasso, Elastic Net, Kernel Ridge, SVR, Gradient Boosting Regression, XGBoost, LightGBM. Finally I got a result 0.1155 of Root-Mean-Squared-Error (RMSE) on logarithm of the house price and ranked top 3% (110/3152) on Kaggle. 

<p align="center">
  <img src="certification2.png" alt="drawing" width="400"/>
</p>



### Usage

OS X & Linux:

```sh
git clone https://github.com/zhangchi9/Ames-Iowa-house-prices-prediction.git
```

Windows:

```sh
git clone https://github.com/zhangchi9/Ames-Iowa-house-prices-prediction.git
```
Run house_price_visualization.ipynb for data visualization. 

Run Model.ipynb for model training. The dataset is already included in the repository, no need to download. 

### Prerequisites

Python 3.6

Jupyter Notebook

### Meta

[Chi Zhang](https://zhangchi9.github.io/) – [@LinkedIn](https://www.linkedin.com/in/chi-zhang-2018/) – c.zhang@neu.edu

Distributed under the MIT license. See [LICENSE](https://github.com/zhangchi9/Ames-Iowa-house-prices-prediction/blob/master/LICENSE) for more information.


<!-- Markdown link & img dfn's -->
[mit-url]:https://opensource.org/licenses/MIT
[mit-image]:https://img.shields.io/badge/License-MIT-yellow.svg
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki
