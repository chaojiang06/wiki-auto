var data = {}; // this is the comprehensive doc, read in line by line

var article_1_para_list = [];
var article_2_para_list = [];
var article_1_id_list = []
var article_2_id_list = []

var partial_aligned_corresponding_list = {};
var aligned_corresponding_list = {};

var aligned_id_list = []
var partial_aligned_id_list = []

var line_num_1 = []

var FLAG_document_is_ready = false;
var FLAG_current_id = false;

function clearChildNodes(id) {
  var myNode = document.getElementById(id);
  while (myNode.firstChild) {
      myNode.removeChild(myNode.firstChild);
  }
}

function add(a, b) {
    return a + b;
}

function onlyUnique(value, index, self) { 
    return self.indexOf(value) === index;
}

function alignment_file_path_load(eve){
  document.getElementById("submit_change_button").classList.remove("bold")
  document.getElementById("submit_change_button").classList.remove("red")
  document.getElementById("save_file_button").classList.remove("bold")
  document.getElementById("save_file_button").classList.remove("red")

  make_load_button_red_bold()

  clearChildNodes("article_name");
  clearChildNodes("article_1_show");
  clearChildNodes("article_2_show");
  data = {}; // this is the comprehensive doc, read in line by line
  article_1_para_list = [];
  article_2_para_list = [];
  article_1_id_list = []
  article_2_id_list = []
  line_num_1 = []

  partial_aligned_corresponding_list = {};
  aligned_corresponding_list = {};
  aligned_id_list = []
  partial_aligned_id_list = []

  FLAG_document_is_ready = false;
  FLAG_current_id = false;

  file = eve.files[0];
  reader = new FileReader();
  
  reader.onload = function(progressEvent){
    var lines = this.result.split('\n');
    line_num_1 = lines[0].split('|')

    for(var i = 1; i < lines.length; i++){
      if (lines[i] == "") {continue;}

      var line = lines[i].split('|')
      var tmp = line[0].slice(3);
      for (var j = 0; j < 6; j++){
        var n = tmp.lastIndexOf("-");
        tmp = tmp.slice(0, n)
      }
      if (!(tmp in data)){data[tmp] = [];}
      data[tmp].push(line)      
    }
    populate_dropdown_list();
  };
  reader.readAsText(file);
}

function populate_dropdown_list(){
  clearChildNodes("article_name");
  var article_name_list = Object.keys(data)
  for (var i = 0; i < article_name_list.length; i++){
    var parent = document.getElementById("article_name");
    var para = document.createElement("option");
    para.text = article_name_list[i];
    para.value = article_name_list[i];
    parent.appendChild(para);
  }
}

function set_article_name_and_level(){

  document.getElementById("submit_change_button").classList.remove("bold")
  document.getElementById("submit_change_button").classList.remove("red")
  document.getElementById("save_file_button").classList.remove("bold")
  document.getElementById("save_file_button").classList.remove("red")
  
  clearChildNodes("article_1_show");
  clearChildNodes("article_2_show");
  article_1_para_list = [];
  article_2_para_list = [];
  article_1_id_list = []
  article_2_id_list = []

  partial_aligned_corresponding_list = {};
  aligned_corresponding_list = {};

  aligned_id_list = []
  partial_aligned_id_list = []

  FLAG_current_id = false;
  FLAG_document_is_ready = false;

  var e = document.getElementById("article_name");
  var article_name = e.options[e.selectedIndex].value;


  var e = document.getElementById("article_1_level");
  var article_1_level = e.options[e.selectedIndex].value;

  var e = document.getElementById("article_2_level");
  var article_2_level = e.options[e.selectedIndex].value;
  
  if (parseInt(article_1_level) <= parseInt(article_2_level)){confirm("Article 1 readability should be larger (Article 1 should be simpler than Article 2)."); return;}

  load_articles(article_name, article_1_level, article_2_level);
  console.log("A");
  FLAG_document_is_ready = true;

  document.getElementById("load_file_button").classList.remove("bold")
  document.getElementById("load_file_button").classList.remove("red")
}

function make_load_button_red_bold(){
  document.getElementById("load_file_button").classList.add("bold")
  document.getElementById("load_file_button").classList.add("red")
}

function load_articles(article_name, article_1_level, article_2_level){
  var current_article = data[article_name];
  for (var i = 0; i < current_article.length; i++){
    var sent_article_name;

    var sent_1_level;
    var sent_2_level;

    var sent_1_para_id;
    var sent_2_para_id;

    var sent_1_sent_id;
    var sent_2_sent_id;

    var sent_1_id;
    var sent_2_id;

    var tmp = current_article[i][0].slice(3);
      for (var j = 0; j < 6; j++){
        var n = tmp.lastIndexOf("-");
        if (j == 0){
          sent_2_sent_id = parseInt(tmp.slice(n+1));
        } else if (j == 1) {
          sent_2_para_id = parseInt(tmp.slice(n+1));
        } else if (j == 2) {
          sent_2_level = parseInt(tmp.slice(n+1));
        } else if (j == 3) {
          sent_1_sent_id = parseInt(tmp.slice(n+1));
        } else if (j == 4) {
          sent_1_para_id = parseInt(tmp.slice(n+1));
        } else if (j == 5) {
          sent_1_level = parseInt(tmp.slice(n+1));
        }
        tmp = tmp.slice(0, n);
        sent_article_name = tmp;
      }

    var sent_1_content = current_article[i][3];
    var sent_2_content = current_article[i][5];

    sent_1_id = article_name+'-'+sent_1_level+'-'+sent_1_para_id+'-'+sent_1_sent_id;
    sent_2_id = article_name+'-'+sent_2_level+'-'+sent_2_para_id+'-'+sent_2_sent_id;

    if (article_1_level == sent_1_level) {
      if (typeof(article_1_para_list[sent_1_para_id]) ==  "undefined" ) {
        article_1_para_list[sent_1_para_id] = [];
      }
      article_1_para_list[sent_1_para_id][sent_1_sent_id] = sent_1_content;
    }

    if (article_2_level == sent_2_level) {
      if (typeof(article_2_para_list[sent_2_para_id]) ==  "undefined" ) {
        article_2_para_list[sent_2_para_id] = [];
      }
      article_2_para_list[sent_2_para_id][sent_2_sent_id] = sent_2_content;
    }



    if (parseInt(current_article[i][6]) == '2'){
      if (typeof(partial_aligned_corresponding_list[sent_1_id]) == 'undefined'){
        partial_aligned_corresponding_list[sent_1_id] = [];
      }
      
      if (!partial_aligned_corresponding_list[sent_1_id].includes(sent_2_id)){
        partial_aligned_corresponding_list[sent_1_id].push(sent_2_id)
      }

    } else if (parseInt(current_article[i][6]) == '1') {
      if (typeof(aligned_corresponding_list[sent_1_id]) == 'undefined'){
        aligned_corresponding_list[sent_1_id] = [];
      }

      if (!aligned_corresponding_list[sent_1_id].includes(sent_2_id)){
        aligned_corresponding_list[sent_1_id].push(sent_2_id)
      }
    }
  }

  var parent = document.getElementById("article_1_show");
  
  for (var i = 0; i < article_1_para_list.length; i++){
    for (var j = 0; j < article_1_para_list[i].length; j++){
      var para = document.createElement("span");
      var t = document.createTextNode(" "+article_1_para_list[i][j]);
      para.setAttribute("id", article_name+"-"+article_1_level+"-"+i+"-"+j);
      para.style.backgroundColor = "lavender"
      para.setAttribute("onclick", "left_click(this.id); return false;");
      para.appendChild(t);
      parent.appendChild(para);
      article_1_id_list.push(article_name+"-"+article_1_level+"-"+i+"-"+j)
      
    }
    if (i != article_1_para_list.length - 1){
      var para = document.createElement("br");
      parent.appendChild(para);
      var para = document.createElement("br");
      parent.appendChild(para);
    }
  }
  var parent = document.getElementById("article_2_show");

  for (var i = 0; i < article_2_para_list.length; i++){
    for (var j = 0; j < article_2_para_list[i].length; j++){
      var para = document.createElement("span");
      var t = document.createTextNode(" "+article_2_para_list[i][j]);
      para.setAttribute("id", article_name+"-"+article_2_level+"-"+i+"-"+j);
      para.style.backgroundColor = "lavenderblush"
      para.setAttribute("onclick", "right_click(this.id); return false;");
      para.appendChild(t);
      parent.appendChild(para);
      article_2_id_list.push(article_name+"-"+article_2_level+"-"+i+"-"+j)

    }
    if (i != article_2_para_list.length - 1){
      var para = document.createElement("br");
      parent.appendChild(para);
      var para = document.createElement("br");
      parent.appendChild(para);
    }
  }
}

function removeEleFromArr(arr, ele){
  var new_arr = []
  for (var i = 0; i<arr.length; i++){
    if (arr[i] != ele){new_arr.push(arr[i])}
  }
  return new_arr
}

function right_click(id){
  if (FLAG_document_is_ready == false) {return;}

  var current_active_left_node = []
  for (var i=0; i < article_1_id_list.length; i++){
      if (document.getElementById(article_1_id_list[i]).style.backgroundColor == "orange")
      {current_active_left_node.push(article_1_id_list[i])}
  }
  if (current_active_left_node.length == 0) {return}

  document.getElementById("submit_change_button").classList.add("bold")
  document.getElementById("submit_change_button").classList.add("red")


  var click_right_node = document.getElementById(id)
  if (click_right_node.style.backgroundColor == "lavenderblush"){click_right_node.style.backgroundColor = "yellow"; partial_aligned_id_list.push(id);}
  else if (click_right_node.style.backgroundColor == "yellow"){click_right_node.style.backgroundColor = "orange"; aligned_id_list.push(id); partial_aligned_id_list = removeEleFromArr(partial_aligned_id_list, id);}
  else if (click_right_node.style.backgroundColor == "orange"){click_right_node.style.backgroundColor = "lavenderblush"; aligned_id_list = removeEleFromArr(aligned_id_list, id);}
}



function close_all_others(){
  aligned_id_list = []
  partial_aligned_id_list = []

  for (var i=0; i < document.getElementById("article_1_show").childNodes.length; i++){
    if (document.getElementById("article_1_show").childNodes[i].tagName == "SPAN")
    {document.getElementById("article_1_show").childNodes[i].style.backgroundColor = "lavender"}
  }

  for (var i=0; i < document.getElementById("article_2_show").childNodes.length; i++){
    if (document.getElementById("article_2_show").childNodes[i].tagName == "SPAN")
    {document.getElementById("article_2_show").childNodes[i].style.backgroundColor = "lavenderblush"}
  }
}

function left_highlight(id){
  aligned_id_list = aligned_corresponding_list[id]
  partial_aligned_id_list = partial_aligned_corresponding_list[id]

  if (typeof(partial_aligned_id_list) == "undefined"){partial_aligned_id_list = []};
  if (typeof(aligned_id_list) == "undefined"){aligned_id_list = []};
  
  var left_node = document.getElementById(id);
  left_node.style.backgroundColor = "orange";
  left_node.scrollIntoView({
    behavior: 'auto',
    block: 'center',
    inline: 'center'
  })

  if (typeof(aligned_id_list) != "undefined") {
    for (var i = 0; i < aligned_id_list.length; i++){
      var right_node = document.getElementById(aligned_id_list[i]);
      if (right_node!= null){
        right_node.style.backgroundColor = "orange";
        right_node.scrollIntoView({
          behavior: 'auto',
          block: 'center',
          inline: 'center'
        })
      }
    }
  }
  
  if (typeof(partial_aligned_id_list) != "undefined") {
    for (var i = 0; i < partial_aligned_id_list.length; i++){
      var right_node = document.getElementById(partial_aligned_id_list[i]);
      console.log(right_node)
      if (right_node!= null){
        right_node.style.backgroundColor = "yellow";
        right_node.scrollIntoView({
          behavior: 'auto',
          block: 'center',
          inline: 'center'
        })
      }
    }
  }
}

function left_click(id) {
  if (FLAG_document_is_ready == false) {return;}
  FLAG_current_id = id
  if (document.getElementById(id).style.backgroundColor == "lavender"){
    close_all_others();
    left_highlight(id);
  } else{
    close_all_others();
  }
  console.log(id, document.getElementById(id),document.getElementById(id).style.backgroundColor == "lavender")
}

function get_next_id(tmp_current_id){
  for (var j = 0; j < 2; j++){
        var n = tmp_current_id.lastIndexOf("-");
        if (j == 0){
          small_sent_id = parseInt(tmp_current_id.slice(n+1));
        } else if (j == 1) {
          small_para_id = parseInt(tmp_current_id.slice(n+1));
        } 
        tmp_current_id = tmp_current_id.slice(0, n);
      }
  
   if (article_1_para_list[small_para_id].length >  small_sent_id){
     return (tmp_current_id+"-"+small_para_id+"-"+(small_sent_id+1))
   } else if (small_para_id + 1 < article_1_para_list.length){
     return tmp_current_id+"-"+(small_para_id + 1)+"-"+'0'
   } else {
     return tmp_current_id+"-"+small_para_id+"-"+small_sent_id
   }
}


function get_previous_id(tmp_current_id){
  for (var j = 0; j < 2; j++){
        var n = tmp_current_id.lastIndexOf("-");
        if (j == 0){
          small_sent_id = parseInt(tmp_current_id.slice(n+1));
        } else if (j == 1) {
          small_para_id = parseInt(tmp_current_id.slice(n+1));
        } 
        tmp_current_id = tmp_current_id.slice(0, n);
      }

   if (small_sent_id >  0){
     return (tmp_current_id+"-"+small_para_id+"-"+(small_sent_id-1))
   } else if (small_para_id >0){
     return (tmp_current_id+"-"+(small_para_id - 1)+"-"+article_1_para_list[small_para_id - 1]);
   } else {
     return (tmp_current_id+"-"+small_para_id+"-"+small_sent_id);
   }
} 

function get_article_name_from_node_id(tmp_current_id){
    for (var j = 0; j < 3; j++){
    var n = tmp_current_id.lastIndexOf("-");
    if (j == 0){
      small_sent_id = parseInt(tmp_current_id.slice(n+1));
    } else if (j == 1) {
      small_para_id = parseInt(tmp_current_id.slice(n+1));
    } 
    tmp_current_id = tmp_current_id.slice(0, n);
  }

  return tmp_current_id
}


function submit_change(){
  document.getElementById("save_file_button").classList.add("bold")
  document.getElementById("save_file_button").classList.add("red")

  FLAG_document_is_ready = false;
  
  var current_active_left_node = []
  for (var i=0; i < article_1_id_list.length; i++){
      if (document.getElementById(article_1_id_list[i]).style.backgroundColor == "orange")
      {current_active_left_node.push(article_1_id_list[i])}
  }

  if (current_active_left_node.length === 1){
    current_active_left_node = current_active_left_node[0]
    aligned_corresponding_list[current_active_left_node] = aligned_id_list
    partial_aligned_corresponding_list[current_active_left_node] = partial_aligned_id_list
  

    var current_article_name = get_article_name_from_node_id(current_active_left_node)
    console.log(current_article_name)
    for (var i = 0; i<data[current_article_name].length; i++){
      var tmp = data[current_article_name][i][0].slice(3);
      for (var j = 0; j < 6; j++){
        var n = tmp.lastIndexOf("-");
        if (j == 0){
          sent_2_sent_id = parseInt(tmp.slice(n+1));
        } else if (j == 1) {
          sent_2_para_id = parseInt(tmp.slice(n+1));
        } else if (j == 2) {
          sent_2_level = parseInt(tmp.slice(n+1));
        } else if (j == 3) {
          sent_1_sent_id = parseInt(tmp.slice(n+1));
        } else if (j == 4) {
          sent_1_para_id = parseInt(tmp.slice(n+1));
        } else if (j == 5) {
          sent_1_level = parseInt(tmp.slice(n+1));
        }
        tmp = tmp.slice(0, n);
        sent_article_name = tmp;
      }

      sent_1_id = sent_article_name+'-'+sent_1_level+'-'+sent_1_para_id+'-'+sent_1_sent_id;
      sent_2_id = sent_article_name+'-'+sent_2_level+'-'+sent_2_para_id+'-'+sent_2_sent_id;

      if (sent_1_id == current_active_left_node){
        if (aligned_id_list.includes(sent_2_id)){data[current_article_name][i][6] = 1} 
        else if (partial_aligned_id_list.includes(sent_2_id)){data[current_article_name][i][6] = 2}
        else {data[current_article_name][i][6] = 3}
      }
    }
  }
  FLAG_document_is_ready = true;
  document.getElementById("submit_change_button").classList.remove("bold")
  document.getElementById("submit_change_button").classList.remove("red")
}

function concate_arrays(){
  var new_arr = []
  new_arr.push(line_num_1)
  for (var key in data) {
    if (data.hasOwnProperty(key)) {
        for (var i = 0; i< data[key].length;i++){new_arr.push(data[key][i])}
    }
  }
  return new_arr
}

function exportToCsv(filename, rows) {
    var processRow = function (row) {
        var finalVal = "";
        for (var j = 0; j < row.length; j++) {
            var innerValue = row[j] === null ? '' : row[j].toString();
            if (row[j] instanceof Date) {
                innerValue = row[j].toLocaleString();
            };
            var result = innerValue;

            if (j > 0)
                finalVal += '|';
            finalVal += result;
        }
        return finalVal + '\n';
    };

    var csvFile = '';
    for (var i = 0; i < rows.length; i++) {
        csvFile += processRow(rows[i]);
    }

    var blob = new Blob([csvFile], { type: 'text/csv;charset=utf-8;' });
    if (navigator.msSaveBlob) { // IE 10+
        navigator.msSaveBlob(blob, filename);
    } else {
        var link = document.createElement("a");
        if (link.download !== undefined) { // feature detection
            // Browsers that support HTML5 download attribute
            var url = URL.createObjectURL(blob);
            link.setAttribute("href", url);
            link.setAttribute("download", filename);
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }
}

function save_file(){



  FLAG_document_is_ready = false;
  const rows = concate_arrays()
  console.log(rows.length, rows[0].length)

  var today = new Date();
  download_file_name = today.getFullYear()+'_'+(today.getMonth()+1)+'_'+today.getDate()+"_"+today.getHours() + "_" + today.getMinutes() + "_" + today.getSeconds();
  
  exportToCsv(download_file_name, rows)

  FLAG_document_is_ready = true;

  document.getElementById("save_file_button").classList.remove("bold")
  document.getElementById("save_file_button").classList.remove("red")

}

document.onkeydown=function(e){    
  var keyNum=window.event ? e.keyCode :e.which;   
  if (FLAG_document_is_ready == true){
    var current_active = false;

    for (var i=0; i < article_1_id_list.length; i++){
      if (document.getElementById(article_1_id_list[i]).style.backgroundColor == "orange")
      {current_active = i}
    }


    if (current_active === false && FLAG_current_id != false){console.log("X");left_click(FLAG_current_id)}
    if (current_active === false && FLAG_current_id == false){console.log("Y");left_click(document.getElementById("article_1_show").childNodes[0].id)}

    if (current_active !== false){

      if(keyNum==38){  
        if (typeof(article_1_id_list[current_active-1])!="undefined"){left_click(article_1_id_list[current_active-1])}
        else {left_click(article_1_id_list[current_active])}
      }  

      if(keyNum==40){  
        console.log(current_active, article_1_id_list[current_active+1])
        if (typeof(article_1_id_list[current_active+1])!="undefined"){left_click(article_1_id_list[current_active+1])}
        else {left_click(article_1_id_list[current_active])}
      }  
    }
  }
}