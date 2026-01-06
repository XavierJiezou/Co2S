from pyecharts import options as opts
from pyecharts.charts import Radar
from pyecharts.commons.utils import JsCode

# ==========================================
# 1. 颜色配置
# ==========================================
color_map = {
    "Co2S (Ours)": "#6c5ce7",
    "OnlySup": "#00cec9",
    "FixMatch": "#fdcb6e",
    "U2PL": "#e17055",
    "WSCL": "#00b894",
    "UniMatch": "#e84393",
    "DWL": "#0984e3",
    "MUCA": "#d63031"
}

# ==========================================
# 2. 数据定义
# ==========================================
data_co2s = [[62.2, 62.7, 79.9, 75.4, 60.9, 56.8]]
data_muca = [[60.0, 60.2, 76.0, 67.4, 52.4, 52.6]]
data_dwl = [[57.1, 62.8, 79.8, 72.6, 59.0, 54.5]]
data_unimatch = [[60.4, 60.0, 74.8, 73.9, 57.9, 56.0]]
data_wscl = [[59.8, 59.4, 73.9, 72.8, 58.2, 51.9]]
data_u2pl = [[59.9, 59.6, 73.8, 67.2, 55.5, 54.9]]
data_fixmatch = [[59.8, 60.1, 73.5, 70.0, 56.8, 53.7]]
data_onlysup = [[58.0, 56.6, 70.2, 67.5, 53.6, 52.7]]

# ==========================================
# 3. Schema 定义
# ==========================================
indicators = ["WHDLD", "LoveDA", "Potsdam", "GID-15", "MSL", "MER"]

schema = [
    opts.RadarIndicatorItem(name=indicators[0], max_=63, min_=45),
    opts.RadarIndicatorItem(name=indicators[1], max_=64, min_=45),
    opts.RadarIndicatorItem(name=indicators[2], max_=81, min_=45),
    opts.RadarIndicatorItem(name=indicators[3], max_=77, min_=45),
    opts.RadarIndicatorItem(name=indicators[4], max_=62, min_=45),
    opts.RadarIndicatorItem(name=indicators[5], max_=58, min_=45),
]

# ==========================================
# 4. 彻底修复的 JS Formatter (单行模式)
# ==========================================
# 关键修改：将所有 JS 代码压缩为一行，避免换行符引发的 SyntaxError
js_str = (
    "function(params) {"
    "var n=['WHDLD','LoveDA','Potsdam','GID-15','MSL','MER'];"
    "var h='<div style=\"font-size:14px;color:#333;font-weight:bold;margin-bottom:5px\">'+params.name+'</div>';"
    "h+='<table style=\"width:100%;font-size:13px;color:#666;\">';"
    "for(var i=0;i<params.value.length;i++){"
    "var v=params.value[i].toFixed(1);"
    "h+='<tr>';"
    "h+='<td style=\"padding:2px 5px;\"><span style=\"display:inline-block;width:8px;height:8px;border-radius:50%;background-color:'+params.color+'\"></span></td>';"
    "h+='<td style=\"padding:2px 5px;\">'+n[i]+'</td>';"
    "h+='<td style=\"padding:2px 5px;text-align:right;font-weight:bold;color:#333;\">'+v+'</td>';"
    "h+='</tr>';"
    "}"
    "h+='</table>';"
    "return h;"
    "}"
)
tooltip_js = JsCode(js_str)

# ==========================================
# 5. 构建图表
# ==========================================
c = (
    Radar(init_opts=opts.InitOpts(width="100%", height="550px", bg_color="white"))
    .add_schema(
        schema=schema,
        shape="polygon",
        center=["42%", "50%"],
        radius="80%",
        splitarea_opt=opts.SplitAreaOpts(
            is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1, color=["#ffffff", "#f9f9f9"])
        ),
        splitline_opt=opts.SplitLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(color="#dcdcdc")),
        axisline_opt=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color="#b0b0b0")),
        textstyle_opts=opts.TextStyleOpts(color="#2d3436", font_size=14, font_weight="bold"),
    )
    .add("Co2S (Ours)", data_co2s, color=color_map["Co2S (Ours)"], symbol="circle",
         linestyle_opts=opts.LineStyleOpts(width=4), areastyle_opts=opts.AreaStyleOpts(opacity=0.2))
    .add("OnlySup", data_onlysup, color=color_map["OnlySup"], symbol="circle",
         linestyle_opts=opts.LineStyleOpts(width=2), areastyle_opts=opts.AreaStyleOpts(opacity=0.1))
    .add("FixMatch (NeurIPS'20)", data_fixmatch, color=color_map["FixMatch"], symbol="circle",
         linestyle_opts=opts.LineStyleOpts(width=2), areastyle_opts=opts.AreaStyleOpts(opacity=0.1))
    .add("U2PL (CVPR'22)", data_u2pl, color=color_map["U2PL"], symbol="circle",
         linestyle_opts=opts.LineStyleOpts(width=2), areastyle_opts=opts.AreaStyleOpts(opacity=0.1))
    .add("WSCL (TGRS'23)", data_wscl, color=color_map["WSCL"], symbol="circle",
         linestyle_opts=opts.LineStyleOpts(width=2), areastyle_opts=opts.AreaStyleOpts(opacity=0.1))
    .add("UniMatch (CVPR'23)", data_unimatch, color=color_map["UniMatch"], symbol="circle",
         linestyle_opts=opts.LineStyleOpts(width=2), areastyle_opts=opts.AreaStyleOpts(opacity=0.1))
    .add("DWL (ISPRS'24)", data_dwl, color=color_map["DWL"], symbol="circle",
         linestyle_opts=opts.LineStyleOpts(width=2), areastyle_opts=opts.AreaStyleOpts(opacity=0.1))
    .add("MUCA (TGRS'25)", data_muca, color=color_map["MUCA"], symbol="circle",
         linestyle_opts=opts.LineStyleOpts(width=2), areastyle_opts=opts.AreaStyleOpts(opacity=0.1))
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        legend_opts=opts.LegendOpts(
            type_="scroll",
            pos_left="76%",
            pos_top="middle",
            orient="vertical",
            item_width=30,
            item_height=16,
            item_gap=15,
            textstyle_opts=opts.TextStyleOpts(
                font_size=16,
                font_family="Arial",
                color="#2d3436"
            )
        ),
        tooltip_opts=opts.TooltipOpts(
            trigger="item",
            formatter=tooltip_js,  # 使用单行 JS 代码
            background_color="rgba(255, 255, 255, 0.95)",
            border_width=1,
            border_color="#eee",
            textstyle_opts=opts.TextStyleOpts(color="#333"),
        ),
    )
)

output_file = "radar_chart_final.html"
c.render(output_file)
print(f"成功生成无错文件：{output_file}")
