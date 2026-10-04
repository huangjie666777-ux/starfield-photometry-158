# 星场定位 + 差分孔径测光后端

从零实现的星场定位 (plate solving) 与差分孔径测光后端:接收二维 FITS
图像与 ICRS 星表,不使用原 WCS 或在线服务。定位输出配对、残差、RMS 及
TAN 投影 WCS,并可下载写入新 WCS 的 FITS;测光端对 2–20 张同滤镜已校正
帧逐帧复用定位流程,输出经参考星零点校正的目标星光变曲线与 CSV。

差分测光用于**区分目标光变与帧间透明度变化**:每帧独立用有效参考星拟合
零点(公共透明度/消光被零点吸收),目标星永不参与校准,因此其自身亮度
变化保留,云量/消光等共模变化被去除。

## 环境

- Python 3.10.12,FastAPI 0.115.12,Astropy 6.1.7,NumPy 2.2.6(见 requirements.lock)
- 所有命令使用 .venv/bin/python

## 模块划分(跨文件协作)

- app/extract.py — 中位数背景 + MAD 噪声估计;阈值成图、连通域检测、
  扣背景加权质心。排除非有限、饱和、边缘不完整源及孤立热像素;
  阈值 threshold_sigma 与饱和值 saturation 可按请求配置。
- app/match.py — 星表按 TAN(gnomonic)投影到切平面(角秒);拒绝距投影
  中心 >=90 度(背向)的星。三角形不变量(边长比,尺度/镜像无关)+ 变换参数
  投票做几何匹配,与列表顺序无关,支持旋转、平移、镜像;允许漏检、假源、
  视场外星;最终一对一贪心指派。搜索预算受 MATCH_TIME_BUDGET_S 限制,
  耗尽时明确报错。
- app/fit.py — 对一致配对最小二乘拟合仿射 pixel = A*tangent + b,
  MAD 迭代剔除离群并重拟合;要求 >=6 对且非共线(SVD 检查);
  最终 RMS(角秒)超过调用者 rms_max 则失败,不交付 WCS。
- app/wcsbuild.py — 由仿射逆构造 TAN WCS(CD 矩阵,度/像素);
  内部像素为零起点,FITS CRPIX 为一基参考点(已 +1 转换);
  清除旧 WCS 关键字(CRPIX/CRVAL/CD1_1/PC1_1/CDELT/CROTA/CTYPE/PV1_2 等
  严格按 FITS WCS 语法匹配),保留像素值与其他观测头(含 PC_TEMP 这类以
  PC 开头的观测关键字);不做任何畸变项。
- app/solver.py — 定位共享流程(提取→投影→匹配→稳健拟合→WCS),
  供 /api/solve 与测光端逐帧复用;另提供目录星到像素的投影。
- app/aperture.py — 差分孔径测光:像素中心落入圆孔径即求和;
  背景环内剔除其他星源污染与非有限/饱和像素后用中位数 + MAD sigma
  clipping 稳健估计局部背景;误差含源+背景泊松、读出噪声及背景均值
  估计不确定度。输出扣背景通量率(e-/s)与误差。
- app/calibrate.py — 每帧参考星零点 ZP = mag + 2.5log10(flux),
  按传播误差加权,MAD 两轮 (4sigma->3sigma) 稳健剔除异常参考星;
  不足 3 颗则该帧失败;目标星等 = ZP - 2.5log10(flux_rate),误差传播。
- app/lightcurve.py — 多帧编排:校验 DATE-OBS 与正 EXPTIME(UTC
  曝光中点 MJD),逐帧定位、测光、校准;单帧失败仅记录原因,其余继续;
  结果按曝光中点排序。
- app/main.py — FastAPI 入口、参数校验、资源限制、JSON 与 CSV 交付。
- app/models.py — 请求/响应模型与服务器端限制
  (图像 <=4096x4096 像素、星表 <=2000 颗、FITS <=64MB 等)。

## API

### POST /api/solve (multipart/form-data)

- image: 二维 FITS 文件(主 HDU 含图像数据)
- params: JSON 字符串,例如:
  {"catalog": [{"id": "star001", "ra": 150.01, "dec": 20.02}, ...],
   "center_ra": 150.0, "center_dec": 20.0,
   "pixel_scale_min": 1.0, "pixel_scale_max": 1.6,
   "rms_max": 0.5, "threshold_sigma": 5.0, "saturation": 60000, "max_pairs": 200}

  pixel_scale_* 单位为角秒/像素;rms_max 单位角秒。
  非法参数(尺度范围倒置、星表 ID 重复、越界坐标、超限规模、
  背向投影中心的星)返回 422/413。

成功返回: solve_id、n_pairs、rms_arcsec、mirrored、
pairs(星表 ID、零起点像素质心、残差角秒)及 wcs(TAN 头关键字)。
配对不足、几何退化或 RMS 超标时返回 422 且不交付 WCS。

### GET /api/solve/{solve_id}/fits

下载写入新 WCS 的 FITS:像素值与其他观测头原样保留,旧 WCS 已清除。

### POST /api/photometry (multipart/form-data)

测量变星亮度并校正帧间透明度。输入:

- images: **2–20 张**同滤镜、已校正二维 FITS(主 HDU 图像);
- params: JSON 字符串,在定位参数之外另含:
  - target_id: 目标星 ID(必须在 catalog 中,且不得是参考星);
  - references: 至少 3 颗 id/mag/mag_err 已知星等参考星,ID 必须存在于
    catalog、不重复、不等于目标;
  - aperture_radius / inner_radius / outer_radius:孔径与背景环半径(像素),
    强制 aperture_radius < inner_radius <= outer_radius;
  - gain(e-/ADU,>0)、read_noise(e-/像素,>=0);
  - 每帧头必须有合法 DATE-OBS(ISO/UTC;也支持 MJD-OBS)与正数值 EXPTIME。

测光规则:逐帧复用同一套定位流程,目录星投到像素坐标;像素中心落入孔径
即求和。出现以下任一情况的星或帧标注原因且**不输出星等**:非有限像素、
饱和(像素 >= saturation)、孔径越界、拥挤(其他星进入其 2 倍孔径范围或
污染背景环)、非正净通量、定位失败、有效参考星不足 3 颗。

成功返回按 MJD 排序的帧记录:mjd_mid(UTC 曝光中点)、flux_rate 与
flux_rate_err(e-/s,含泊松+读出+背景估计噪声)、target_mag 与
target_mag_err、zero_point 与 zero_point_err、n_refs_used,以及每颗参考星
的采用/排除状态及原因(如 zero-point outlier、saturated、crowded)。
单帧失败时保留该帧记录(ok=false、reason),其他帧继续处理。

### GET /api/photometry/{photometry_id}/csv

下载与 JSON 完全一致的 CSV(一行一帧,含采用/排除参考星及质量原因),
结果在内存中保留,服务重启后失效。

## 运行与验证

    .venv/bin/python -m pytest tests -q          # 合成星场端到端测试
    .venv/bin/python -m compileall -q app tests  # 编译检查
    .venv/bin/python -m uvicorn app.main:app --port 8152

    curl -F "image=@field.fits" -F 'params={"catalog":[...],...}' \
         http://127.0.0.1:8152/api/solve
    curl -OJ http://127.0.0.1:8152/api/solve/<solve_id>/fits

测光(多个 images 表单字段;params 为 JSON 文本字段):

    curl http://127.0.0.1:8152/api/photometry \
      -F "images=@f0.fits" -F "images=@f1.fits" -F "images=@f2.fits" \
      -F "params=@photo.json;type=application/json"
    curl -OJ http://127.0.0.1:8152/api/photometry/<photometry_id>/csv

photo.json 示例(定位参数 + 测光参数):

    {
      "catalog": [{"id": "star001", "ra": 150.01, "dec": 20.02}],
      "center_ra": 150.0, "center_dec": 20.0,
      "pixel_scale_min": 1.0, "pixel_scale_max": 1.6, "rms_max": 0.5,
      "target_id": "star000",
      "references": [
        {"id": "star001", "mag": 12.30, "mag_err": 0.01}],
      "aperture_radius": 4.0,
      "inner_radius": 7.0, "outer_radius": 12.0,
      "gain": 1.8, "read_noise": 7.0
    }

光变曲线示例(目标 +/-0.35 等正弦光变 + 帧间 60%-100% 透明度变化,
8 帧;先启动服务再运行):

    .venv/bin/python examples/variable_star_demo.py --port 8152

示例打印恢复出的差分星等与逐帧零点,并写出 photometry_demo.csv:
零点随透明度变化,目标星等恢复出原始正弦幅度。

## 测试范围

tests/test_solver.py 生成合成星场(高斯星点 + 噪声),覆盖:

- 任意旋转(35、80、123 度)与平移的求解;
- 镜像视场(mirrored=true);
- 漏检星(随机丢弃)、假源(视场外噪声峰)、孤立热像素排除;
- 拒绝背向投影中心的星表星;拒绝不可能达到的 RMS 门槛;
- 拒绝非法参数(尺度范围倒置);
- 下载 FITS 的 WCS 回算天区坐标与星表一致(<1 角秒),观测头保留。
- 星表乱序(打破“前 30 项”列表顺序)不影响定位结果(顺序依赖修复);
- PC_TEMP 等观测头在 WCS 重写后保留(PC 前缀误删修复)。

tests/test_photometry.py 在合成光变序列上覆盖:

- 目标 +/-0.3 等正弦光变、帧间 0.6-1.0 透明度变化下的恢复与零点跟踪;
- 结果按 MJD 排序、每帧不少于 3 颗参考星、CSV 与 JSON 一致;
- 单帧目标孔径内 NaN:该帧记录失败原因(不输出星等),其余帧继续;
- EXPTIME=0 的帧失败并保留记录;
- 半径次序、gain、目标-参考星关系等参数 422 校验。

## 范围说明

- 仅支持二维主 HDU 图像;不支持多扩展、压缩图像或畸变模型。
- 投影为标准 TAN(gnomonic),无视场畸变改正。
- 测光输入要求为**已校正**(bias/dark/flat 后)同滤镜帧;不做仪器
  特征去除、颜色项/大气质量改正,不做多滤镜交叉校准。
- 孔径与背景环半径由调用者给定(像素);不做 PSF 拟合或变孔径优化。
- 求解/测光结果保存在内存中,服务重启后 ID 失效。
- 匹配预算固定为 20 秒;超大规模或极端密度星场可能因预算耗尽而失败。
- 测光最多 20 帧;图像规模限制与定位相同(<=4096x4096 像素等)。
