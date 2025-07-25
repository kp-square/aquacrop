# --- Detailed NRCS Type II 24-Hour Rainfall Distribution Data (Tabular) ---
# Source: https://www.hydrocad.net/rainfall/tables/Type%20II%2024-hr%20Tabular.hcr

NRCS_Type2_Table_Data = """
depth=.0000  .0010  .0020  .0030  .0041  .0051  .0062  .0072  .0083  .0094
depth=.0105  .0116  .0127  .0138  .0150  .0161  .0173  .0184  .0196  .0208
depth=.0220  .0232  .0244  .0257  .0269  .0281  .0294  .0306  .0319  .0332
depth=.0345  .0358  .0371  .0384  .0398  .0411  .0425  .0439  .0452  .0466
depth=.0480  .0494  .0508  .0523  .0538  .0553  .0568  .0583  .0598  .0614
depth=.0630  .0646  .0662  .0679  .0696  .0712  .0730  .0747  .0764  .0782
depth=.0800  .0818  .0836  .0855  .0874  .0892  .0912  .0931  .0950  .0970
depth=.0990  .1010  .1030  .1051  .1072  .1093  .1114  .1135  .1156  .1178
depth=.1200  .1222  .1246  .1270  .1296  .1322  .1350  .1379  .1408  .1438
depth=.1470  .1502  .1534  .1566  .1598  .1630  .1663  .1697  .1733  .1771
depth=.1810  .1851  .1895  .1941  .1989  .2040  .2094  .2152  .2214  .2280
depth=.2350  .2427  .2513  .2609  .2715  .2830  .3068  .3544  .4308  .5679
depth=.6630  .6820  .6986  .7130  .7252  .7350  .7434  .7514  .7588  .7656
depth=.7720  .7780  .7836  .7890  .7942  .7990  .8036  .8080  .8122  .8162
depth=.8200  .8237  .8273  .8308  .8342  .8376  .8409  .8442  .8474  .8505
depth=.8535  .8565  .8594  .8622  .8649  .8676  .8702  .8728  .8753  .8777
depth=.8800  .8823  .8845  .8868  .8890  .8912  .8934  .8955  .8976  .8997
depth=.9018  .9038  .9058  .9078  .9097  .9117  .9136  .9155  .9173  .9192
depth=.9210  .9228  .9245  .9263  .9280  .9297  .9313  .9330  .9346  .9362
depth=.9377  .9393  .9408  .9423  .9438  .9452  .9466  .9480  .9493  .9507
depth=.9520  .9533  .9546  .9559  .9572  .9584  .9597  .9610  .9622  .9635
depth=.9647  .9660  .9672  .9685  .9697  .9709  .9722  .9734  .9746  .9758
depth=.9770  .9782  .9794  .9806  .9818  .9829  .9841  .9853  .9864  .9876
depth=.9887  .9899  .9910  .9922  .9933  .9944  .9956  .9967  .9978  .9989
depth=1.000"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def get_array(table):
    data = []
    x_split = table.split('depth=')
    for lin in x_split:
        items = lin.split()
        if items:
            data.extend(float(val) for val in items)
    return data


def nrcs_type2_hourly_dissociation(daily_rainfall) -> list:
    """
    Disassociates daily rainfall to hourly rainfall using NRCS type II distribution.
    Args:
        daily_rainfall: daily_rainfall data (floating point number)

    Returns:
        A list of size 24 with each value representing estimated rainfall for that hour.
    """
    data = get_array(NRCS_Type2_Table_Data)
    cumulative_data = [daily_rainfall * val for val in data]
    incremental = [cumulative_data[i] - cumulative_data[i - 1] for i in range(1, len(cumulative_data))]
    hourly_rainfall = [round(sum(incremental[i: i + 10]), 2) for i in range(0, len(incremental), 10)]
    return hourly_rainfall

def irrigation_dissociation(daily_irrigation) -> list:
    """
    Disassociates irrigation to hourly to avoid sharp wetting front in soil for stability of richard equation
    solution. Irrigation is applied from 5 am to 10 am (5 hours).
    Args:
        daily_irrigation
    Returns:
        A list of size 24 with each value representing estimated irrigation for that hour.
    """
    n = 5
    r = 2.5
    total_weight = sum(r ** i for i in range(n))
    a = daily_irrigation / total_weight
    diss = [a * r**i for i in range(n)]
    diss = [0]*5 + diss
    diss.extend([0] * (24 - len(diss)))
    return diss