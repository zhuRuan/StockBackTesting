# UFA 量化 SDK

## 配置环境

本 SDK 使用 Python 3 语言。我们推荐您使用 Python 3.8 或以上版本。

## 用途

- `apis/`
  - 用于获取行情数据(`finance_data.py`)和交易数据(`trade.py`)的 API
- `strategy/`
  - 量化策略脚本
  - 具体结构请参考策略样例(`example_strat.py`)
- `utils/`
  - 工具函数
  - 可根据需求使用`market_tools.py`中定义的工具函数，其余文件请忽略
- `config.py`
  - 配置文件
- `run_strategy.py`
  - 程序入口

## 运行前配置

请在`config.py`中配置以下变量：

- `API_KEY`: UFA 平台中提供的 API KEY，用于后台辨认身份
- `STRATEGY_NAME`: 需要运行的策略的文件名（例：若需要运行我们提供的策略样例`strategy/example_strat.py`，请将此变量设为`"example_strat"`)
- `STRATEGY_INTERVAL`: 策略运行间隔（分钟），需大于等于 1，否则可能运行失败

## 运行

- 自定义策略
  - 在`strategy/`下新建一个 python 文件，文件名无特殊要求
  - 复制策略样例中的代码并粘贴至新文件中
  - 将`main()`函数中的代码替换为您自定义的策略代码
  - 若需要执行此策略，在`config.py`中将`STRATEGY_NAME`修改为此文件的文件名，并按需求调整运行间隔`STRATEGY_INTERVAL`
- 执行策略
  - 执行`run_strategy.py`并保持运行
    - 可在命令行中执行`python run_strategy.py`
    - 也可使用 IDE 提供的执行功能
  - 运行中修改策略
    - 若在运行中发现策略需要修改，可直接修改对应的策略脚本并保存
    - SDK 在运行中将监听并自动应用修改过的策略，因此您在修改时**不需要切断正在运行的脚本**
