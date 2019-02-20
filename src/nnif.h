#ifndef __nnif__
#define __nnif__
#include "nnbase.h"
#include <string>
#include <map>
#include <deque>

typedef struct
{
  int       src;
  int       src2;
  int       target;
  double    from;
  double    from2;
  double    through;
  double    through2;
  double    step;
  double    step2;
  long      start;
  long      len;
  long      skip;
  void     *mod;
  double    val;
  double    avg;
  double    curr;
  double    conf;
  char      avg_ind;
  char      curr_ind;
} ANAVALS;


class nnif
{
  public :
   typedef enum {save_arch, save_patterns, save_model} SAVE_TYPE; 
   typedef map <int, double> NVALUE;
   typedef map <int, string> SVALUE;
   typedef struct {
     NVALUE  inp;  // the list of inputs
     NVALUE  out;  // the list of outputs
     double  err;  // reported error on pattern
   } PATTERN;
   
  private :
   typedef struct {
     int layer;    // connect to layer
     int node;     // connect to specific node if >= 0, else to whole layer
   } CONNSPEC;
   
   typedef pair<CONNSPEC, CONNSPEC> CONNPAIR;

   typedef struct {
     string            text;         // layer text labels
     int               type;         // 0 = middle, 1 = input, 2 = output
     int               ncount;       // nodes in this layer
   } LAYER;
   
   network             *m_net;
   map   <int, PATTERN> m_pattern; 
   map   <int, LAYER>   m_layer;
   deque <CONNPAIR>     m_connect;
   PATTERN              m_test;
   string               m_params;
   SVALUE               m_ilab;           // input labels
   SVALUE               m_olab;           // output labels
   NVALUE               m_imin, m_imax;   // min/max values for inputs
   NVALUE               m_min, m_max;     // min/max values for outputs
   bool                 m_random;
   bool                 m_batchmode;
   double               m_avg_err;
   ANAVALS              m_anavals;
   
   void check_layer(int n) {
     if (m_layer.find(n)==m_layer.end()) {
       m_layer[n].text = string(""); m_layer[n].ncount = 0;
     }
   }
   
  public :
   nnif();
   
   // minimums/maximums for inputs/outputs
   double min(int n) { 
     if (m_min.find(n)==m_min.end()) return 0.0; return m_min[n]; }
   double max(int n) { 
     if (m_max.find(n)==m_max.end()) return 1.0; return m_max[n]; }
   double imin(int n) { 
     if (m_imin.find(n)==m_imin.end()) return 0.0; return m_imin[n]; }
   double imax(int n) { 
     if (m_imax.find(n)==m_imax.end()) return 1.0; return m_imax[n]; }
   
   int layers() { return m_layer.size(); }     // number of layers
   int patterns() { return m_pattern.size(); } // number of patterns

   // Text getters/setters
   const char *inp_text(int inp_no) { 
     if (inp_no < no_inputs()) return m_ilab[inp_no].c_str(); return ""; }
   void inp_text(int inp_no, const char *a_txt) { 
     if (inp_no < no_inputs()) m_ilab[inp_no] = string(a_txt); }
     
   const char *out_text(int out_no) { 
     if (out_no < no_outputs()) return m_olab[out_no].c_str(); return ""; }
   void out_text(int out_no, const char *a_txt) { 
     if (out_no < no_outputs()) m_olab[out_no] = string(a_txt); }
   
   const char *layertext(int a_lno) { 
     if (a_lno < layers()) return m_layer[a_lno].text.c_str(); return "";
   }
   
   void layertext(int a_lno, const char *a_text) { 
     if (a_text && a_text[0]) {
       check_layer(a_lno); m_layer[a_lno].text = string(a_text);
     }
   }
   const char *nodelabel(int lay, int n) { 
     if (m_net) return m_net->tag(lay, n); return "";
   }
   // getting and setting input/output patterns
   void input(const char *a_val, int pattern, int n) {
     m_pattern[pattern].inp[n] = atof(a_val);
   };
   void input(double a_val, int pattern, int n) {
     m_pattern[pattern].inp[n] = a_val;
   };
   void output(const char *a_val, int pattern, int n) {
     m_pattern[pattern].out[n] = atof(a_val);
   };
   void output(double a_val, int pattern, int n) {
     m_pattern[pattern].out[n] = a_val;
   };
   
   double input(int pattern, int n) {return m_pattern[pattern].inp[n]; }
   double output(int pattern, int n) { return m_pattern[pattern].out[n]; }
   
   void ltypeof(int layer, int type) {
     if (layer < layers()) {       
       if (layer==0) m_layer[layer].type=1;
       else if (layer+1==layers()) m_layer[layer].type=2;
       else m_layer[layer].type=type;
     }     
   };
   int  ltypeof(int layer) {
     if (layer<0) return 0;
     if (layer==0) return 1;
     if (layer+1==layers()) return 2;
     if (layer < layers()) return m_layer[layer].type;
     return 0;
   };
   int nodecount(int layer){
     if (layer < layers()) return m_layer[layer].ncount;
     return 0;
   }
   void nodecount(int layer, int count){
     if (layer < layers()) m_layer[layer].ncount = count; 
   }
   
   int no_inputs() { return nodecount(0); }
   
/*   int no_inputs() {
     if (layers() > 0) {
       
       int c = 0;
       connection *l_row;
       if (m_net)
	 for (int lay = 0; lay < layers(); lay++)
	   if (ltypeof(lay)==1) 
	     if ((l_row = m_net->row(lay))!=NULL) c += l_row->count();
       return c;
     }
     return 0;
   }
*/
   
   int no_outputs() {
     if (layers()) return m_layer[layers()-1].ncount; return 0; 
   }
   const double avg_error() { return m_avg_err; }
   void connect(int layera, int layerb){
     CONNPAIR l_con;
     if (layera==layerb) return;
     l_con.first.layer = layera; l_con.second.layer = layerb;
     l_con.first.node = -1;      l_con.second.node = -1;
     m_connect.push_back(l_con);
   };
   void connect(int layera, int layerb, int node){
     CONNPAIR l_con;
     if (layera==layerb) return;
     l_con.first.layer = layera; l_con.second.layer = layerb;
     l_con.first.node  = -1;     l_con.second.node = node;
     m_connect.push_back(l_con);
   };
   void connect(int layera, int nodea, int layerb, int nodeb){
     CONNPAIR l_con;
     if (layera==layerb) return;
     l_con.first.layer = layera; l_con.second.layer = layerb;
     l_con.first.node  = nodea;  l_con.second.node  = nodeb;
     m_connect.push_back(l_con);
   };
   void disconnect(int layera, int layerb);
   void disconnect(int layera, int layerb, int nodeb);
   void disconnect(int layera, int nodea, int layerb, int nodeb);
   
   string &params() { return m_params; }
   string param(const string &a_name) {
     if (!m_net) return string("");
     return m_net->param(a_name.c_str());
   }   
   void params(const char *a_type, double eta, double lambda) {
     char l_params[128];
     sprintf(l_params, "type=%s eta=%g lambda=%g", a_type, eta, lambda);
//     printf("setting params to %s\n", l_params);
     m_params = string(l_params);
     if (m_net) m_net->params(m_params.c_str()); 
   }
   void params(const char *a_type, double eta, double lambda, double decay) {
     char l_params[128];
     sprintf(l_params, "type=%s eta=%g lambda=%g decay=%g", 
	     a_type, eta, lambda, decay);
     m_params = string(l_params);
//     printf("setting params to %s\n", l_params);
     if (m_net) m_net->params(m_params.c_str()); 
   }
   void params(const char *a_type, double eta, double lambda, double decay,
	       double kappa, double theta, double maxeta) {
     char l_params[128];
     sprintf(l_params, 
	     "type=%s eta=%g lambda=%g decay=%g kappa=%g theta=%g etamax=%g", 
	     a_type, eta, lambda, decay, kappa, theta, maxeta);
//     printf("setting params to %s\n", l_params);
     m_params = string(l_params);
     if (m_net) m_net->params(m_params.c_str()); 
   };
   
   void reset(){
     delete m_net; m_net = new network();
     m_ilab.clear(); m_olab.clear();
     m_imin.clear(); m_imax.clear();
     m_min.clear();  m_max.clear();
     m_pattern.clear();
     m_layer.clear();
     m_connect.clear();
     m_params = "";
     m_avg_err = 0;
   }
   void     sequential(bool a_random) { 
     m_batchmode = false; m_random = !a_random; }
   void     randomise(bool a_random) { 
     m_batchmode = false; m_random = a_random; }
   void     batchmode(bool a_batch) { m_random=false; m_batchmode = a_batch; }
   void     load(const char *a_fname, SAVE_TYPE a_fmt);
   void     save(const char *a_fname, SAVE_TYPE a_fmt);
   void     build();       // rebuild the network, using this configuration
   double   teach();       // present next pattern set and learn. return error
   PATTERN &query(int a_patno);  // present pattern set and return values
   void     disp();
   void     analyse();
   void     get_anavals(double &avg, double &curr, double &conf, double &val,
			int &avg_ind, int &curr_ind)
   {
     avg = m_anavals.avg;
     curr = m_anavals.curr;
     conf = m_anavals.conf;
     avg_ind = m_anavals.avg_ind;
     curr_ind = m_anavals.curr_ind;
     val = m_anavals.val;
   }
   
   void anavalues(int src, int target, 
		  double from, double through, double step,
		  double start, double len, double skip);
   void anavalues(int id1, int id2, int target,
		  double from1, double through1, double step1,
		  double from2, double through2, double step2,
		  int patno);
   
   char *report_arch();
   char *report_setup();
};
#endif
