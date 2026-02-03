"use strict";
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
            tabActive: "0",
            tabActiveName: "产品体系",
            listLableMap: ["产品体系", "个人理财", "机构理财"],
            //与v-model中的必须保持一致
            tableData: [],
            searchValue: "",
            riskGrage: "不限", // 风险评级
            termType: "不限", // 产品期限
            prdcTyp: "不限", // 产品类型
            prdcStt: "不限", // 产品状态
            prdcFrm: "不限", // 产品形态
            sllObjc: "无", // 是否个人
            currentPage: 1,
            searchword: "",
            getSearchTotal: [],
            totalNum: 1,
            getPageIndex: null,
            getPageTotal: null,
            pageNow: 1, //默认显示第几页数据
            pageSize: 10, //这里是设置每一页显示多少条数据
            pageSizeNew: 10, //这里是设置每一页显示多少条数据
            getRiskGrage: {
                不限: "不限",
                R1: "低风险",
                R2: "较低风险",
                R3: "中等风险",
                R4: "较高风险",
                R5: "高风险",
            },
            getpPrdcStt: {
                募集: "募集",
                存续类: "存续",
            },
            pageJson: []
        },
        mounted: function mounted() {
            if (this.tabActive != '0') {
                this.initset()
            }
        },
        methods: {
            openInfo: function (value, index) {
                this.tabActive = index;
                this.tabActiveName = value;
                this.activeName = value;
                if (value != '0') {
                    this.initset()
                }
            },
            // 筛选是否显示字段
            handelIsShow: function (params) {
                // $.ajax({
                //   type: "get",
                //   url: "https://www.spdb-wm.com/financialProducts/NAV/NAV2.json",
                //   // data: data,
                //   dataType: "json",
                //   async: false,
                //   cache: false,
                //   success: function (res) {
                //     if (res && res.data.length > 0) {
                //       for (let index = 0; index < res.data.length; index++) {
                //         const element = res.data[index];
                //         for (let index1 = 0; index1 < params.length; index1++) {
                //           const tadaInfo = params[index1];
                //           if (element.REAL_PRD_CODE == tadaInfo.PRDC_CD) {
                //             if (element.TOT_NAV == "0E-8") {
                //               tadaInfo.TOT_NAV = 0;
                //             } else {
                //               tadaInfo.TOT_NAV = element.TOT_NAV;
                //             }
                //           }
                //         }
                //       }
                //     } else {

                //     }
                //   },
                //   error: function (e) {

                //   },
                // });
                // const arr = [ 
                //   2301202803,
                //   2301202810,
                //   2301202813,
                //   2301203615,
                //   2301202810,
                //   2301203612,
                //   2301213712,
                //   2301213744,
                //   2301213746,
                //   2301220040,
                //   2301229030,
                //   2301229031,
                //   2301229032,
                //   2301229042,
                //   2301221302,
                //   2301220807,
                //   2301229029,
                //   2301221003,
                //   2301229033,
                //   2301229034,
                //   2301229035,
                //   2301229045,
                //   2301229047,
                //   2301229049,
                //   2301220969,
                //   2301229040,
                //   2301229050,
                //   2301229051,
                //   2301221026,
                //   2301220818,
                //   2301220819,
                //   2301229052,
                //   2301220935,
                //   2301220831,
                //   2301220940,
                //   2301221319,
                //   2301221031,
                //   2301221037,
                //   2301220730,
                //   2301220845,
                //   2301220738,
                //   2301220956,
                //   2301221322,
                //   2301221325,
                //   2301221042,
                //   2301220850,
                //   2301220754,
                //   2301220755,
                //   2301221332,
                //   2301221333,
                //   2301220830,
                //   2301221055,
                //   2301221062,
                //   2301220776,
                //   2301221068,
                //   2301221093,
                //   2301230800,
                //   2301230801,
                //   2301230940,
                //   2301230946,
                //   2301241501,
                //   2501240050
                // ];
                return params.filter(item => {
                    // let str = Number(item.PRDC_CD)
                    if (item.SHR != '' && item.SHR != 0 && item.SHR != "0E-8") {
                        if (item.FML_PRDC_IDNT == '是' && item.FML_PRDC_IDNT_2 == 'N') {
                            return true
                        } else {
                            if (item.FML_PRDC_IDNT == '否') {
                                return true
                            } else {
                                return false
                            }
                        }
                    } else {
                        return false
                    }


                    // if(  ( item.FML_PRDC_IDNT == '是' && item.FML_PRDC_IDNT_2 == 'N' ) &&   ) {
                    //   return false;
                    // } else {
                    //   return true;
                    // }
                    // if( item.YLD_7 || item.TDY_MLLN_CPS_PRFT  ) {
                    //   return true
                    // } else {
                    //   if( item.TOT_NAV ) {
                    //     return true
                    //   } else  {
                    //     return false
                    //   }
                    // }
                });
            },
            initset: function initset() {
                this.riskGrage = "不限";
                this.termType = "不限";
                this.prdcTyp = "不限";
                this.prdcStt = "不限";
                this.prdcFrm = "不限";
                this.searchValue = "";
                this.pageNow = 1;
                this.initData(1);
            },
            indexMethod: function indexMethod(index) {
                return index + 1 < 10 ? "0".concat(index + 1) : index + 1;
            },
            handleCurrentChange: function handleCurrentChange(val) {
                console.log("\u5F53\u524D\u9875: ".concat(val));
                this.pageNow = val;
                this.currentPage = val;
                this.initData(val);
            },
            groupChange: function groupChange(val) {
                this.pageNow = 1;
                this.initData(1);
                this.currentPage = 1;
            },
            /**
             * @description: 处理数据
             * @param {*} page
             * @return {*}
             */
            handleFilter: function (data, type, tag) {
                if (
                    (tag == "RISK_GRADE" && data === "不限") ||
                    (tag == "TERM_TYPE" && data === "不限") ||
                    (tag == "PRDC_TYP" && data === "不限") ||
                    (tag == "PRDC_STT" && data === "不限") ||
                    (tag == "PRDC_FRM" && data === "不限") ||
                    (tag == "SLL_OBJC" && data === "无")
                ) {
                    return true;
                } else {
                    if (data === type) {
                        return true;
                    } else {
                        if (tag == "RISK_GRADE") {
                            return data == type ? true : false;
                        } else {
                            return data ? type.indexOf(data) != -1 : true;
                        }
                    }
                }
            },
            initData: function (page) {
                var then = this;
                var p = page || 1;
                var then = this;
                // 拼接筛选项
                // 筛选项list
                let searchwordList = []
                // 筛选项枚举
                let searchList = {
                    riskGrage: 'RISK_GRADE',
                    termType: 'TERM_TYPE',
                    prdcTyp: 'PRDC_TYP',
                    prdcStt: 'PRDC_STT',
                    prdcFrm: 'PRDC_FRM',
                }
                // 将筛选的添加到list中
                for (const key in searchList) {
                    if (this[key] !== '不限' && key === 'riskGrage') {
                        searchwordList.push(`${searchList[key]}='${this.getRiskGrage[`${this.riskGrage}`]}'`)
                    } else if (this[key] !== '不限') {
                        searchwordList.push(`${searchList[key]}='${this[key]}'`)

                    }
                }
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
                        size: 99999,
                        searchword: searchwordList.length > 0 ? `(${searchwordList.join(' and ')})` : ''
                    }),
                    contentType: 'application/json',
                    async: true,
                    cache: false,
                    success: function (resData) {
                        let res = {
                            data: resData.data.content
                        }
                        if (res && res.data.length > 0) {
                            var getData = [];
                            for (let index = 0; index < res.data.length; index++) {
                                const item = res.data[index];
                                const procStt = item.PRDC_STT;
                                if (procStt.indexOf('募集') != -1 || procStt.indexOf('存续') != -1) {
                                    getData.push({
                                        ...item,
                                        PRDC_NM: item.PRDC_NM + "/" + item.PRDC_CD
                                    })
                                }
                            }
                            //去重,并且保留没有子级产品的父级产品
                            getData = then.filterProductDataSimple(getData);
                            //去重,并且保留没有子级产品的父级产品
                            getData = then.noRepeatHandle(getData);
                            // 去除( 含有累计净值字段并且值为0)的数据
                            getData = then.handelIsShow(getData);
                            then.initTotNav(getData);

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

            noRepeatHandle: function (data) {
                const keptParentCodes1 = new Set();
                return data.filter(item => {
                    if (keptParentCodes1.has(item.PRDC_CD)) {
                        return false
                    } else {
                        keptParentCodes1.add(item.PRDC_CD)
                        return true
                    }
                })
            },

            filterProductDataSimple: function (data) {
                // 获取所有子级产品的父级码值
                const parentCodesWithChildren = new Set(
                    data
                    .filter(item => item.FML_PRDC_CD !== item.PRDC_CD)
                    .map(item => item.FML_PRDC_CD)
                );

                const keptParentCodes = new Set();

                // 筛选：保留所有子级产品 + 没有子级产品的父级产品
                return data.filter(item => {
                    // 如果是子级产品，直接保留
                    if (item.FML_PRDC_CD !== item.PRDC_CD) {
                        return true;
                    }
                    // 如果是父级产品，检查是否有子级产品
                    if (!parentCodesWithChildren.has(item.PRDC_CD)) {
                        if (keptParentCodes.has(item.PRDC_CD)) {
                            return false
                        } else {
                            keptParentCodes.add(item.PRDC_CD);
                            return true
                        }
                    }
                    return false;
                });
            },

            initTotNav: function (params) {
                var then = this;
                $.ajax({
                    type: "POST",
                    url: "https://www.spdb-wm.com/api/search",
                    data: JSON.stringify({
                        chlid: 1006,
                        cutsize: 150,
                        dynexpr: [],
                        dynidx: 1,
                        extopt: [],
                        orderby: '',
                        page: 1,
                        size: 99999,
                        searchword: ''
                    }),
                    contentType: 'application/json',
                    async: true,
                    cache: false,
                    success: function (resData) {
                        let res = {
                            data: resData.data.content
                        }
                        if (res && res.data.length > 0) {
                            var array = [];
                            for (let index = 0; index < res.data.length; index++) {
                                const element = res.data[index];
                                for (let index1 = 0; index1 < params.length; index1++) {
                                    const tadaInfo = params[index1];
                                    if (element.REAL_PRD_CODE == tadaInfo.PRDC_CD) {
                                        if (element.TOT_NAV == "0E-8") {
                                            tadaInfo.TOT_NAV = 0;
                                        } else {
                                            tadaInfo.TOT_NAV = element.TOT_NAV;
                                        }
                                        if (element.NAV == "0E-8") {
                                            tadaInfo.NAV = 0;
                                        } else {
                                            tadaInfo.NAV = element.NAV;
                                        }
                                        tadaInfo.ISS_DATE = element.ISS_DATE;
                                    } else {
                                        // tadaInfo.NAV = "";
                                        // tadaInfo.TOT_NAV = "";
                                        // tadaInfo.ISS_DATE = "";
                                    }
                                }
                            }
                            let searchData = params.filter(function (item, inde, array) {
                                return (
                                    then.handleFilter(
                                        then.getRiskGrage[then.riskGrage],
                                        item.RISK_GRADE,
                                        "RISK_GRADE"
                                    ) &&
                                    then.handleFilter(
                                        then.termType,
                                        item.TERM_TYPE,
                                        "TERM_TYPE"
                                    ) &&
                                    then.handleFilter(then.prdcTyp, item.PRDC_TYP, "PRDC_TYP") &&
                                    then.handleFilter(
                                        then.getpPrdcStt[then.prdcStt],
                                        item.PRDC_STT,
                                        "PRDC_STT"
                                    ) &&
                                    then.handleFilter(then.prdcFrm, item.PRDC_FRM, "PRDC_FRM") &&
                                    then.handleFilter(then.sllObjc, item.SLL_OBJC, "SLL_OBJC") &&
                                    (then.handleFilter(then.searchValue, item.PRDC_NM, "") ||
                                        then.handleFilter(then.searchValue, item.PRDC_CD, "")) &&
                                    (then.handleFilter(then.tabActive == '1' ? "对私" : "对公", item.SLL_OBJC, "SLL_OBJC") ||
                                        then.handleFilter("对公,对私", item.SLL_OBJC, "SLL_OBJC"))
                                );
                            });
                            then.totalNum = searchData.length; // 获取总数
                            then.getPageTotal = Math.round(searchData.length / 10); // 获取总条数
                            // 做筛选
                            then.tableData = searchData.splice(
                                (then.pageNow - 1) * then.pageSizeNew,
                                then.pageSizeNew
                            );
                            then.$forceUpdate();
                        } else {
                            // then.tableData = [];
                            // then.totalNum = 0;
                        }
                    },
                    error: function (e) {
                        then.tableData = [];
                        (then.currentPage = 0), (then.totalNum = 0);
                    },
                });
            },
            JumpPage: function (params) {
                if (params.row.RS_MTHD == "私募") {} else {
                    var url =
                        "https://www.spdb-wm.com/financialProducts/cpxq.shtml?REAL_PRD_CODE=" +
                        params.row.PRDC_CD;
                    window.open(url, "_black");
                }
            },
        },
    });
})();