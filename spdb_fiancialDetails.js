"use strict";

function _typeof(obj) {
  "@babel/helpers - typeof";
  return (
    (_typeof =
      "function" == typeof Symbol && "symbol" == typeof Symbol.iterator
        ? function (obj) {
          return typeof obj;
        }
        : function (obj) {
          return obj &&
            "function" == typeof Symbol &&
            obj.constructor === Symbol &&
            obj !== Symbol.prototype
            ? "symbol"
            : typeof obj;
        }),
    _typeof(obj)
  );
}
function ownKeys(object, enumerableOnly) {
  var keys = Object.keys(object);
  if (Object.getOwnPropertySymbols) {
    var symbols = Object.getOwnPropertySymbols(object);
    enumerableOnly &&
      (symbols = symbols.filter(function (sym) {
        return Object.getOwnPropertyDescriptor(object, sym).enumerable;
      })),
      keys.push.apply(keys, symbols);
  }
  return keys;
}
function _objectSpread(target) {
  for (var i = 1; i < arguments.length; i++) {
    var source = null != arguments[i] ? arguments[i] : {};
    i % 2
      ? ownKeys(Object(source), !0).forEach(function (key) {
        _defineProperty(target, key, source[key]);
      })
      : Object.getOwnPropertyDescriptors
        ? Object.defineProperties(
          target,
          Object.getOwnPropertyDescriptors(source)
        )
        : ownKeys(Object(source)).forEach(function (key) {
          Object.defineProperty(
            target,
            key,
            Object.getOwnPropertyDescriptor(source, key)
          );
        });
  }
  return target;
}
function _defineProperty(obj, key, value) {
  key = _toPropertyKey(key);
  if (key in obj) {
    Object.defineProperty(obj, key, {
      value: value,
      enumerable: true,
      configurable: true,
      writable: true,
    });
  } else {
    obj[key] = value;
  }
  return obj;
}
function _toPropertyKey(arg) {
  var key = _toPrimitive(arg, "string");
  return _typeof(key) === "symbol" ? key : String(key);
}
function _toPrimitive(input, hint) {
  if (_typeof(input) !== "object" || input === null) return input;
  var prim = input[Symbol.toPrimitive];
  if (prim !== undefined) {
    var res = prim.call(input, hint || "default");
    if (_typeof(res) !== "object") return res;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (hint === "string" ? String : Number)(input);
}
(function () {
  // 检查是否需要 Polyfill
   if (!Array.prototype.filter) {
     Array.prototype.filter = function (func, thisArg) {
       "use strict";
       if (
         !((typeof func === "function" || typeof func === "Function") && this)
       ) {
         throw new TypeError();
       }

       var len = this.length >>> 0,
         res = new Array(len),
         t = this,
         c = 0,
         i = -1;

       var kValue;
       if (thisArg === undefined) {
         while (++i !== len) {
           if (i in this) {
             kValue = t[i];
             if (func(t[i], i, t)) {
               res[c++] = kValue;
             }
           }
         }
       } else {
         while (++i !== len) {
           if (i in this) {
             kValue = t[i];
             if (func.call(thisArg, t[i], i, t)) {
              res[c++] = kValue;
             }
           }
         }
       }

       res.length = c;
       return res;
     };
   }
  new Vue({
    el: "#container_fian",
    data: {
      tabActive: "1",
      prodTableData: [],
      titleData: [],
      productName: "",
      //与v-model中的必须保持一致
      worthTableData: [],
      currentPage: 1,
      xxplCurrentPage: 1,
      myChart: null,
      totalNum: 1,
      xxplTotalNum: 1,
      getPageIndex: null,
      getPageTotal: null,
      getXxplPageIndex: null,
      getXxplPageTotal: null,
      pageNow: 1, //默认显示第几页数据
      pageSize: 4, //这里是设置每一页显示多少条数据
      pageSizeNew: 4, //这里是设置每一页显示多少条数据
      xxplPageNow: 1, //默认显示第几页数据
      xxplPageSize: 6, //这里是设置每一页显示多少条数据
      xxplPageSizeNew: 6, //这里是设置每一页显示多少条数据
      getEchartsData: [],
      getDStartInfo: {},
      getRiskGrage: {
        R1: "低风险",
        R2: "中低风险",
        R3: "中风险",
        R4: "中高风险",
        R5: "高风险",
      },
      activeName: "0101",
      getIndex: null,
      infoDisclosureList: [], // 信息披露列表
      getPRDCRGSTCD: '',
      isShowDetails:false,
      detailsData:[],
      isShowAll: true,
      loading: false,
      recordData: []
    },
    mounted: function mounted() {
      var _this = this;
      this.initData();
      this.navList1();
      this.yearYield();
      this.initDStart();
      this.getUrlParam("REAL_PRD_CODE");
    },
    methods: {
      returnHref: function (val, event) {
        this.tabActive = val;
        var element = this.$refs[event];
        if (element) {
          element.scrollIntoView({ behavior: "smooth" });
        }
      },
      /**
       * 获取信息披露数据 
       * 
       */
      initInfoDisclosure: function (page) {
        var then = this;
        $.ajax({
          type: "get",
          url: "https://www.spdb-wm.com/financialProducts/XXPL/XXPL.json",
          // data: data,
          dataType: "json",
          async: true,
          cache: false,
          success: function (res) {
            then.getCpsmsData(res);
            // if (res && res.data.length > 0) {
            //   var data = res.data.filter(function (item) { return item.CODE == then.getPRDCRGSTCD && item.TYPE == then.activeName });
            //   then.xxplTotalNum = data.length;
            //   then.infoDisclosureList = data.splice(
            //     (then.xxplPageNow - 1) * then.xxplPageSizeNew,
            //     then.xxplPageSizeNew
            //   );
            //   then.getXxplPageTotal = Math.round(data.length / 6);
            // } else {
            //   then.infoDisclosureList = [];
            //   then.getXxplPageTotal = 0;
            // }
          },
          error: function (e) {
            then.tableData = [];
            (then.xxplCurrentPage = 0), (then.xxplTotalNum = 0);
          },
        });

      },
      getCpsmsData: function(resData) {
        var then = this;
        $.ajax({
          type: "get",
          url: "https://www.spdb-wm.com/financialProducts/cpsms/index.json",
          // data: data,
          dataType: "json",
          async: true,
          cache: false,
          success: function (res) {
            var newArr = resData.data.concat(res.data);
            if (newArr && newArr.length > 0) {
                var data = newArr.filter(function (item) { return item.CODE == then.getPRDCRGSTCD && item.TYPE == then.activeName });
                for (let index = 0; index < data.length; index++) {
                    var element = data[index];
                    var dateStr = element.PUBDATE.split(' ')[0]
                    if (dateStr.indexOf('-') != -1) {
                        // 如果日期字符串包含 -，将其转换为 YYYYMMDD 格式
                        element.PUBDATE = dateStr.replace(/-/g, '');
                    }
                }
                data.sort(function (a, b) {
                  return Number(b.PUBDATE) - Number(a.PUBDATE);
                });
                then.xxplTotalNum = 1;
                then.infoDisclosureList = data.splice(
                    (then.xxplPageNow - 1) * then.xxplPageSizeNew,
                    then.xxplPageSizeNew
                );
                then.getXxplPageTotal = Math.round(data.length / 6);
            } else {
                then.infoDisclosureList = [];
                then.getXxplPageTotal = 0;
            }
          },
          error: function (e) {
            then.tableData = [];
            (then.xxplCurrentPage = 0), (then.xxplTotalNum = 0);
          },
        });
      },
      openFile: function (info) {
        this.getIndex = info;
        var url = '';
        if (info.TYPE && info.attribute) {
          window.open(info.attribute);
        } else if (info.TYPE) {
          window.open('https://www.spdb-wm.com/financialProducts/XXPL/' + info.PDFFILE, "mozillaTab");
        } else {
          window.open(info.HREF, "mozillaTab");
        }
      },
      handleClick: function (tab, name) {
        this.xxplPageNow = 1;
        this.xxplCurrentPage = 1;
        this.getXxplPageTotal = 0;
        this.initInfoDisclosure();
      },
      getUrlParam: function (name) {
        var url = window.location.href;
        var params = url.substr(url.lastIndexOf("?") + 1).split("&");
        for (var i = 0; i < params.length; i++) {
          var param = params[i];
          var key = param.split("=")[0];
          var value = param.split("=")[1];
          if (key === name) {
            return value;
          }
        }
      },
      formatDate: function (dateString) {
        if (dateString != "") {
          var year = dateString.slice(0, 4);
          var month = dateString.slice(4, 6);
          var day = dateString.slice(6, 8);
          var formattedDate = year + '-' + month + '-' + day;
          return formattedDate;
        } else {
          return "-";
        }
      },
      /**
       * @description: 初始化数据
       * @param {*} page
       * @return {*}
       */
      initData: function (page) {
        var then = this;
        var searchword1 = "";
        var p = page || 1;
        var then = this;
        // 获取产品信息
        $.ajax({
          type: "POST",
          url: "https://www.spdb-wm.com/api/search",
          data: JSON.stringify({
            chlid: 1002,
            cutsize: 150,
            dynexpr: [],
            dynidx: 1,
            extopt: [],
            orderby: '',
            page: 1,
            size: 10,
            searchword: `(PRDC_CD = '${then.getUrlParam("REAL_PRD_CODE")}')`
          }),
          contentType: 'application/json',
          dataType: "json",
          async: true,
          cache: false,
          success: function (resData) {
            let res = {
              data: resData.data.content
            }
            if (res && res.data.length > 0) {
              var getData = res.data.filter(function (item) { return item.PRDC_CD == then.getUrlParam("REAL_PRD_CODE") })[0];
              if (getData && getData.length != 0) {
                then.productName = getData.PRDC_NM;
                then.getPRDCRGSTCD = getData.PRDC_RGST_CD;
                then.prodTableData = [
                  {
                    type: "全国银行业理财信息登记系统登记编号",
                    value: getData.PRDC_RGST_CD,
                  },
                  {
                    type: "理财币种",
                    value: getData.RS_CRRN,
                  },
                  {
                    type: "产品类型",
                    value: getData.PRDC_TYP,
                  },
                  {
                    type: "募集方式",
                    value: getData.RS_MTHD,
                  },
                  {
                    type: "运作方式",
                    value: getData.PRDC_FRM,
                  },
                ];
                then.titleData = [
                  {
                    type: "风险评级",
                    value: getData.RISK_GRADE,
                  },
                  {
                    type: "产品名称",
                    value: getData.PRDC_NM,
                  },
                  {
                    type: "产品代码",
                    value: getData.PRDC_CD,
                  },
                ];
                if(getData.YLD_7 || getData.TDY_MLLN_CPS_PRFT){
                  then.detailsData = [
                    {
                      'ACCT_DT': getData.ACCT_DT,
                      'YLD_7': getData.YLD_7,
                      'TDY_MLLN_CPS_PRFT': getData.TDY_MLLN_CPS_PRFT
                    },
                  ];
                }
              }
            } else {
            }
            then.initInfoDisclosure();
          },
          error: function (e) {
            then.tableData = [];
            (then.currentPage = 0), (then.totalNum = 0);
          },
        });
      },
      navList: function (page) {
        var then = this;
        var searchword1 = "";
        var p = page || 1;
        var then = this;
        // 获取产品信息
        $.ajax({
          type: "get",
          url: "https://www.spdb-wm.com/financialProducts/NAV/NAV2.json",
          // data: data,
          dataType: "json",
          async: true,
          cache: false,
          success: function (res) {
            if (res && res.data.length > 0) {
              var getData = res.data.filter(function (item) { return item.REAL_PRD_CODE == then.getUrlParam("REAL_PRD_CODE") });
              getData.sort(function (a, b) {
                return Number(b.ISS_DATE) - Number(a.ISS_DATE);
              });
              for (let index = 0; index < getData.length; index++) {
                var element = getData[index];
                element.ISS_DATE = then.formatDate(
                  element.ISS_DATE,
                  "yyyy-MM-dd"
                );
                if (element.TOT_NAV == "0E-8") {
                  element.TOT_NAV = 0;
                } else { 
                  element.TOT_NAV = element.TOT_NAV;
                }

              }
              then.totalNum = getData.length;
              then.worthTableData = getData.splice(
                (then.pageNow - 1) * then.pageSizeNew,
                then.pageSizeNew
              );
              then.getPageTotal = Math.round(getData.length / 4);
            } else {
              then.tableData = [];
              then.totalNum = 0;
            }
          },
          error: function (e) {
            then.tableData = [];
            (then.currentPage = 0), (then.totalNum = 0);
          },
        });
      },

      lookMoreData: function() {
        this.isShowAll =true;
        this.currentPage = 1;
        this.navList1();
      },

      navList1: function (page) {
        var then = this;
        var searchword1 = "";
        var p = page || 1;
        var then = this;
        then.loading = true
        then.worthTableData = [];
        then.recordData = [];
        // 获取产品信息
        $.ajax({
          type: "POST",
          url: "https://www.spdb-wm.com/api/search",
          data: JSON.stringify({
            chlid: 1003,
            cutsize: 150,
            dynexpr: [],
            dynidx: 1,
            extopt: [],
            orderby: '',
            page: then.currentPage,
            size: 4,
            searchword: `(REAL_PRD_CODE = '${then.getUrlParam("REAL_PRD_CODE")}')`
          }),
          contentType: 'application/json',
          dataType: "json",
          async: true,
          cache: false,
          success: function (resData) {
            let res = {
              data: resData.data.content
            }
            then.loading = false;
            if (res && res.data.length > 0) {
              // var getData = res.data.filter(function (item) { return item.REAL_PRD_CODE == then.getUrlParam("REAL_PRD_CODE") });
              // getData.sort(function (a, b) {
              //   return Number(b.ISS_DATE) - Number(a.ISS_DATE);
              // });
              var getData = res.data
              for (let index = 0; index < getData.length; index++) {
                var element = getData[index];
                element.ISS_DATE = then.formatDate(
                  element.ISS_DATE,
                  "yyyy-MM-dd"
                );
                if (element.TOT_NAV == "0E-8") {
                  element.TOT_NAV = 0;
                } else { 
                  element.TOT_NAV = element.TOT_NAV;
                }

              }
              then.totalNum = +resData.data.totalElements;
              // then.recordData = JSON.parse( JSON.stringify(getData));
              then.worthTableData = getData;
            } else {
              then.tableData = [];
              then.totalNum = 0;
            }
          },
          error: function (e) {
            then.loading = false;
            then.tableData = [];
            (then.currentPage = 0), (then.totalNum = 0);
          },
        });
      },

      yearYield: function () {
        var then = this;
        var searchword1 = "";
        var then = this;
        // 获取产品信息
        $.ajax({
          type: "POST",
          url: "https://www.spdb-wm.com/api/search",
          data: JSON.stringify({
            chlid: 1004,
            cutsize: 150,
            dynexpr: [],
            dynidx: 1,
            extopt: [],
            orderby: '',
            page: 1,
            size: 100,
            searchword: `(PRD_CODE = '${then.getUrlParam("REAL_PRD_CODE")}')`
          }),
          contentType: 'application/json',
          dataType: "json",
          async: true,
          cache: false,
          success: function (resData) {
            let res = {
              data: resData.data.content
            }
            if (res && res.data.length > 0) {
              then.getEchartsData = res.data.filter(function (item) { return item.PRD_CODE == then.getUrlParam("REAL_PRD_CODE") });
              for (let index = 0; index < then.getEchartsData.length; index++) {
                var item = then.getEchartsData[index];
                if (item.YEAR_YIELD == "0E-12") {
                  item.YEAR_YIELD = 0;
                }
                item.percentage = Number(item.YEAR_YIELD).toFixed(2) + "%";
                item.name = item.END_DATE + "年度";
                item.value = Number(item.YEAR_YIELD).toFixed(2);
              }
              then.initSetEcharts();
            } else {
              then.getEchartsData = [];
              // then.totalNum = 0;
            }
          },
          error: function (e) {
            then.tableData = [];
            (then.currentPage = 0), (then.totalNum = 0);
          },
        });
        window.addEventListener("resize", function () {
          then.myChart.resize();
        });
      },
      initDStart: function initDStart() {
        var then = this;
        $.ajax({
          type: "POST",
          url: "https://www.spdb-wm.com/api/search",
          data: JSON.stringify({
            chlid: 1005,
            cutsize: 150,
            dynexpr: [],
            dynidx: 1,
            extopt: [],
            orderby: '',
            page: 1,
            size: 10,
            searchword: `(PRD_CODE = '${then.getUrlParam("REAL_PRD_CODE")}')`
          }),
          contentType: 'application/json',
          dataType: "json",
          async: true,
          cache: false,
          success: function (resData) {
            let res = {
              data: resData.data.content
            }
            for (let index = 0; index < res.data.length; index++) {
              var element = res.data[index];
              if (element.PRD_CODE == then.getUrlParam("REAL_PRD_CODE")) {
                element.STARTDATE = then.formatDate(
                  element.STARTDATE,
                  "yyyy-MM-dd"
                );
                element.END_DATE = then.formatDate(
                  element.END_DATE,
                  "yyyy-MM-dd"
                );
                then.getDStartInfo = element;
              }

            }
          },
          error: function (e) {
            then.tableData = [];
            (then.currentPage = 0), (then.totalNum = 0);
          },
        });
      },
      initSetEcharts: function initSetEcharts() {
        var chartDom2 = document.getElementById("echarts_conents");
        this.myChart = echarts.init(chartDom2);
        if (this.getEchartsData && this.getEchartsData.length > 0) {
          this.getEchartsData.sort(function (a, b) {
            return Number(a.END_DATE) - Number(b.END_DATE);
          });
        }
        var datas = this.getEchartsData;
        var option;
        option = {
          backgroundColor: "#FFF",
          grid: {
            left: 0,
            right: 40,
            bottom: 50,
            top: 40,
            containLabel: true,
          },
          tooltip: {
            trigger: "item",
            axisPointer: {
              type: "none",
            },
            formatter: function formatter(params) {
              return params.data.name + " : " + params.data.value;
            },
          },
          xAxis: {
            show: true,
            type: "category",
            data: datas.map(function (item) {
              return item.name;
            }),
          },
          yAxis: [
            {
              type: "value",
              axisLabel: {
                show: true,
                margin: 10,
                //右侧y轴数字的外边距
                textStyle: {
                  fontSize: 14,
                  color: "rgba(0,0,0,0.5)",
                },
                formatter: function formatter(percentage) {
                  return percentage + "%";
                },
              },
              splitLine: {
                show: true,
                lineStyle: {
                  color: "#DADADA",
                  type: "solid",
                },
              },
              axisTick: {
                show: false,
              },
              axisLine: {
                show: false,
              },
            },
          ],
          series: [
            {
              name: "值",
              type: "bar",
              zlevel: 1,
              showBackground: false,
              label: {
                show: false,
              },
              itemStyle: {
                normal: {
                  color: new echarts.graphic.LinearGradient(0, 1, 0, 0, [
                    {
                      offset: 0,
                      color: "#6094E8",
                    },
                    {
                      offset: 1,
                      color: "#003CC8",
                    },
                  ]),
                },
              },
              barWidth: 24,
              data: datas,
            },
          ],
        };
        option && this.myChart.setOption(option);
      },
      indexMethod: function indexMethod(index) {
        return index + 1 < 10 ? "0".concat(index + 1) : index + 1;
      },
      handleCurrentChange: function handleCurrentChange(val) {
        this.pageNow = val;
        this.currentPage = val;
        this.navList(val);
      },

      handleCurrentChange1: function handleCurrentChange1(val) {
        this.pageNow = val;
        this.currentPage = val;
        this.navList1(val);
        // let result = JSON.parse( JSON.stringify(this.recordData) );
        // this.worthTableData = [];
        // this.$nextTick( () => {
        //   this.worthTableData = result.splice(
        //     (this.pageNow - 1) * this.pageSizeNew,
        //     this.pageSizeNew
        //   );
        // } );
      },

      handleCurrentXxplChange: function handleCurrentXxplChange(val) {
        console.log("\u5F53\u524D\u9875: ".concat(val));
        this.xxplPageNow = val;
        this.xxplCurrentPage = val;
        this.initInfoDisclosure(val);
      },
      headFirst: function headFirst(_ref) {
        var row = _ref.row,
          colunm = _ref.colunm,
          rowIndex = _ref.rowIndex,
          columnIndex = _ref.columnIndex;
        var base = {
          height: "64px",
          "line-height": "37px",
          "background-color": "#003CC8",
          color: "#FFFFFF",
          "font-size": "20px",
          "text-align": "center",
        };
        //这里为了是将第一列的表头隐藏，就形成了合并表头的效果
        if (rowIndex === 0) {
          // 判断对第几列合并  property就是prop传入的属性
          if (row[columnIndex].property === "type") {
            //第一列width扩展2倍
            return _objectSpread(
              _objectSpread({}, base),
              {},
              {
                width: "200%",
                display: "inline-block",
              }
            );
          } else if (row[columnIndex].property === "value") {
            //其余列设置font-size 0 隐藏
            return {
              "font-size": 0,
              display: "none",
            };
          }
        }
        return base;
      },
    },
  });
})();
