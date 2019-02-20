/* An XOR network, using the base neural network class */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include "nnif.h"
#include "ogl_model.h"

nnif::nnif()
{
  m_random = true;
  m_net = new network();
  m_params = "";
  m_anavals.mod = NULL;
}

void nnif::build()
{
  delete m_net;
  m_net = new network(paramlist(m_params.c_str()));
  m_avg_err = 0;
  map <int, LAYER>::iterator layer;
  fprintf(stderr, "\nRebuilding network\n");
  
  for (layer=m_layer.begin(); layer!=m_layer.end(); ++layer)
  {
    if (layer->second.ncount>0)
    {
      fprintf(stderr, "Row %s, %i nodes\n", 
	     layer->second.text.c_str(), layer->second.ncount);      
      m_net->addrow((char *)layer->second.text.c_str(), layer->second.ncount);
    }
    else
    {
      fprintf(stderr, "Freaking out: node count is zero!\n");
      while (m_layer.find(layer->first) != m_layer.end())
	m_layer.erase(layer->first);
      break;
    }
  }
//  printf("\nBuilding connections. Now have %i layers\n", m_layer.size());
  deque <CONNPAIR>::iterator c;
  for (c=m_connect.begin(); c != m_connect.end(); ++c)
  {
    if ((unsigned)c->first.layer>=m_layer.size()) continue;
    connection *l_fconn = m_net->row(c->first.layer);
    if ((unsigned)c->second.layer>=m_layer.size()) continue;
    connection *l_tconn = m_net->row(c->second.layer);
    if (l_fconn==l_tconn)
    {
      fprintf(stderr, "Cannot connect from and to the same layer!\n");
      continue;
    }
    
    if (c->first.node==-1)
    {
      if (c->second.node==-1)
      {
//	fprintf(stderr, "Fully connect %s to %s\n", 
//		l_fconn->tag(),l_tconn->tag());
	m_net->connect_layers(c->first.layer, c->second.layer);
      }      
      else
      {
	int n = c->second.node;
	while (l_tconn && n--) l_tconn = l_tconn->next();
//	fprintf(stderr, "Connect layer %s to node %s\n", 
//		l_fconn->tag(),l_tconn->tag());	
	if (l_tconn) l_fconn->connect_layer_to(paramlist(m_params.c_str()), 
					       l_tconn);
      }      
    }
    else
    {
      if (c->second.node==-1)
      {
	int n = c->first.node;
	while (l_fconn && n--) l_fconn = l_fconn->next();
//	fprintf(stderr, "Connect node %s to layer %s\n", 
//	       l_fconn->tag(),l_tconn->tag());	
	if (l_fconn) l_fconn->connect_to_layer(paramlist(m_params.c_str()), 
					       l_tconn);	
      }
      else  
      {
	int n = c->first.node;
	while (l_fconn && n--) l_fconn = l_fconn->next();
	if (l_fconn) 
	{
	  int n = c->second.node;
	  while (l_tconn && n--) l_tconn = l_tconn->next();	  
//	  fprintf(stderr, "Connect node %s to node %s\n", 
//		  l_fconn->tag(),l_tconn->tag());	
	  if (l_tconn) l_fconn->connect(paramlist(m_params.c_str()), 
					l_tconn);	
	}	
      }      
    }
  }
//  fprintf(stderr, "\nSetting Output to %i\n", m_net->rows()-1);
  if (m_net->rows() > 0) m_net->set_output(m_net->rows()-1);
}

void nnif::disconnect(int layera, int layerb)
{
  deque <CONNPAIR>::iterator c;
  for (c=m_connect.begin(); c != m_connect.end(); ++c)
  {
    if (c->first.layer==layera && c->second.layer==layerb)
    {
      m_connect.erase(c);
      disconnect(layera, layerb);
      return;
    }
  }
}

void nnif::disconnect(int layera, int layerb, int nodeb)
{
  deque <CONNPAIR>::iterator c;
  for (c=m_connect.begin(); c != m_connect.end(); ++c)
  {
    if (c->first.layer==layera && 
	c->second.layer==layerb && c->second.node==nodeb)
    {
      m_connect.erase(c);
      disconnect(layera, layerb, nodeb);
      return;
    }
  }  
}

void nnif::disconnect(int layera, int nodea, int layerb, int nodeb)
{
  deque <CONNPAIR>::iterator c;
  for (c=m_connect.begin(); c != m_connect.end(); ++c)
  {
    if (c->first.layer==layera && c->first.node==nodea &&
	c->second.layer==layerb && c->second.node==nodeb)
    {
      m_connect.erase(c);
      disconnect(layera, nodea, layerb, nodeb);
      return;
    }
  }  
}

char *nnif::report_arch()
{
  static char *l_rep = NULL;
  FILE *l_file = fopen("netmodel.log", "w+");
  if (l_file==NULL) return "Could not open or create status file.\n";
  fprintf(l_file, "Network Architecture:\n");
  fprintf(l_file, "~~~~~~~~~~~~~~~~~~~~~\n");
  fprintf(l_file, "Network has %i layers with %i inputs and %i outputs\n", 
	  layers(), no_inputs(), no_outputs());
  for (int i=1; i+1 < layers(); i++)
    fprintf(l_file, "  Middle layer %i ('%s') has %i nodes\n", 
	    i, layertext(i), nodecount(i));
  fprintf(l_file, "Connections are as follows:\n");
  deque <CONNPAIR>::iterator c;
  for (c=m_connect.begin(); c != m_connect.end(); ++c)
  {  
    char l_node[30];
    
    if (c->first.layer<0 || c->first.node < 0)
      sprintf(l_node, " node %i", c->first.node);
    else if (ltypeof(c->first.layer)==1)
      sprintf(l_node, " node %i(%s)", c->first.node, inp_text(c->first.node));
    else if (ltypeof(c->first.layer)==2)
      sprintf(l_node, " node %i(%s)", c->first.node, out_text(c->first.node));
    else 
      sprintf(l_node, " node %i(%s)", c->first.node, 
	      nodelabel(c->first.layer, c->first.node));
    
    fprintf(l_file, " layer %i(%s)%s connected to ",
	    c->first.layer, layertext(c->first.layer),
	    c->first.node<0?" fully":l_node);
    
    if (c->second.layer<0 || c->second.node < 0)
      sprintf(l_node, " node %i", c->second.node);
    else if (ltypeof(c->second.layer)==1)
      sprintf(l_node, " node %i(%s)", c->second.node, 
	      inp_text(c->second.node));
    else if (ltypeof(c->second.layer)==2)
      sprintf(l_node, " node %i(%s)", c->second.node, 
	      out_text(c->second.node));
    else 
      sprintf(l_node, " node %i(%s)", c->second.node, 
	      nodelabel(c->second.layer, c->second.node));
    
    fprintf(l_file, " layer %i(%s)%s\n", 
	    c->second.layer, layertext(c->second.layer),
	    c->second.node<0?"":l_node);
  }
  SVALUE::iterator s;
  fprintf(l_file, "Input labels are as follows:\n");
  for (s=m_ilab.begin(); s != m_ilab.end(); ++s)
    fprintf(l_file, "  in%-2i=%s\n", s->first, s->second.c_str());
  fprintf(l_file, "Output labels are as follows:\n"); 
  for (s=m_olab.begin(); s != m_olab.end(); ++s)
    fprintf(l_file, "  out%-2i=%s\n", s->first, s->second.c_str());
  
  int l_size = ftell(l_file);
  l_rep = (char *)realloc(l_rep, l_size+1);
  if (l_rep==NULL)
  {
    fclose(l_file);
    return "Error allocating memory for report\n";
  }
  
  l_rep[l_size] = 0;
  fseek(l_file, 0, SEEK_SET);
  fread(l_rep, l_size, 1, l_file);
  fclose(l_file);
  return l_rep;
}

char *nnif::report_setup()
{
  const char *l_bp = 
    "Standard BackPropagation of Error\n"
    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
    " This learning rule is one of the first rules devised\n"
    " capable of learning the 'XOR' problem in supervised\n"
    " feed forward networks.\n"
    " With each iteration, the signal is propagated forward\n"
    " through the network to generate an output signal. \n"
    " The output is compared to the expected output, and \n"
    " the error is propagated back through the network, \n"
    " adjusting weights by a fraction of the error\n"
    " according to each node's contribution to the \n"
    " overall error.\n"
    "Constants:\n"
    "~~~~~~~~~~~\n"
    "Learning rate:\n"
    "\ta multiplier for the error used to adjust the\n"
    "\tweights\n"
    "Momentum Constant: \n"
    "\tthe amount from a previous adjustment which is\n"
    "\tadded to the next weight adjustment\n";
  const char *l_bp_decay = 
    "Standard BackPropagation of Error with Weight Decay\n"
    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
    " Identical to the normal Backpropagation algorithm\n"
    " other than for the addition of a weight decay. Decay\n"
    " effectively stabilises the network by reducing\n"
    " weight contributions steadily with each learning\n"
    " iteration.\n"
    "Constants:\n"
    "~~~~~~~~~~~\n"
    "Learning rate:\n"
    "\ta multiplier for the error used to adjust the\n"
    "\tweights\n"
    "Momentum Constant: \n"
    "\tthe amount from a previous adjustment which is\n"
    "\tadded to the next weight adjustment\n"
    "Decay: \n"
    "\ta small constant by which weights are reduced\n"
    "\twith each learning cycle.";
  const char *l_dbd = 
    "Delta-Bar-Delta\n"
    "~~~~~~~~~~~~~~~\n"
    " A rule with a difference.  Instead of a fixed\n"
    " learning rate, each weight has it's own.  This\n"
    " constant is adjusted downward when weight\n"
    " adjustments appear to be cycling endlessly\n"
    " between positive and negative values.  The\n"
    " learning rate is adjusted upward when the weight\n"
    " adjustments appear to be in the same direction.\n"
    " This rule is incredibly efficient.\n"
    "Constants:\n"
    "~~~~~~~~~~~\n"
    "Learning rate(eta):\n"
    "\ta multiplier for the error used to adjust the\n"
    "\tweights\n"
    "Momentum Constant(lambda): \n"
    "\tthe amount from a previous adjustment which is\n"
    "\tadded to the next weight adjustment\n"
    "Decay: \n"
    "\tThe amount by which the learning rate adjusts\n"
    "\tdownward when weight values cycle.\n"
    "Eta Increment(kappa):\n"
    "\tthe amount by which the weight learning\n"
    "\tis adjusted upward when weight adjustments\n"
    "\tare in a consistent direction.\n"
    "Eta Max:\n"
    "\tThe maximum learning rate for weight eta's\n"
    "Slope Adjustment(theta):\n"
    "\tThe speed with which weights recognise\n"
    "\ta change in direction. Values range between\n"
    "\tzero and 1\n";

  const char *l_bpc = 
    "BackPropagation Counterweights\n"
    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
    " This network topology institutes not one,\n"
    " but two weights between a given node and it's\n"
    " target.  As the network learns, these weights\n"
    " move closer to one another; the distance \n"
    " between the weight and it's counterweight\n"
    " is an indication of error for the connection.\n"
    " This rule has a minimum learning rate and a \n"
    " maximum learning rate. As the distance between\n"
    " the weight and counterweight widens, the effective\n"
    " learning rate draws closer to the minimum.\n"
    " Conversely, a close to zero distance results in a\n"
    " close to maximum learning rate.\n"
    " A momentum constant is provided, although this\n"
    " is not necessary to get a good performance.\n"
    " This rule is almost as efficient as the\n"
    " delta-bar-delta, although it has far fewer\n"
    " parameters, and is rather good at overcoming \n"
    " local minimum problems.\n"
    "Constants:\n"
    "~~~~~~~~~~~\n"
    "Learning rate(eta):\n"
    "\ta multiplier for the error used to adjust\n"
    "\tthe node bias.  Also used as a minimum weight\n"
    "\tadjustment learning rate.\n"
    "Momentum Constant(lambda): \n"
    "\tthe amount from a previous adjustment which is\n"
    "\tadded to the next weight adjustment.  There is\n"
    "\tno need generally for this parameter.\n"
    "Eta Max:\n"
    "\tThe maximum learning rate for any given weight \n"
    "\tadjustment.  This is reached where the weight for\n"
    "\ta synapse is equivalent to it's counterweight."
    "Decay: \n"
    "\tThe amount by which the bias decays with each\n"
    "\tlearning cycle. Clearly, where there is zero bias\n"
    "\tadjustment(eta), this parameter has no effect.\n";
    
  static char *l_rep = NULL;
  FILE *l_file = fopen("netmodel.log", "w+");
  if (l_file==NULL) return "Could not open or create status file.\n";  
  
  if (param("type")==string("bp"))
    fprintf(l_file, "%s", l_bp);  
  else if (param("type")==string("bp_decay"))
    fprintf(l_file, "%s", l_bp_decay);  
  else if (param("type")==string("dbd"))
    fprintf(l_file, "%s", l_dbd);  
  else if (param("type")==string("bpc"))
    fprintf(l_file, "%s", l_bpc);  
  
  int l_size = ftell(l_file);
  l_rep = (char *)realloc(l_rep, l_size+1);
  if (l_rep==NULL)
  {
    fclose(l_file);
    return "Error allocating memory for report\n";
  }
  
  l_rep[l_size] = 0;
  fseek(l_file, 0, SEEK_SET);
  fread(l_rep, l_size, 1, l_file);
  fclose(l_file);
  return l_rep;
}

double nnif::teach()
{
  const double c_residual = 0.99999;
  double l_err = 0.0;
  static int l_pat = 0;
//  if (m_random) l_pat = (int)((double)rand() / (double)RAND_MAX * patterns());
  if (m_random)
  {
    l_pat = 
      (int)(log10(1 + ((double)rand() / (double)RAND_MAX) * 9.0) * patterns());
  }
  
  if (m_net->rows() > 0 && l_pat < patterns())
  {
    int si, so;
    double i[(si = m_pattern[l_pat].inp.size())+1];
    double o[(so = m_pattern[l_pat].out.size())+1];
    for (int n = 0; n < si; n++) i[n] = m_pattern[l_pat].inp[n];
    for (int n = 0; n < so; n++) o[n] = m_pattern[l_pat].out[n];
    int ivals = 0;
    for (int lay = 0; lay < m_net->rows(); lay++)
    {
      if (ltypeof(lay)==1)
      {
	int nvals = m_net->row(lay)->count();
	m_net->set_input(lay, si-ivals, &i[ivals], 0.0, 1.0);
	ivals += nvals;
      }
      
      if (ivals >= si) break;
    }
    
    m_net->set_output(m_net->rows()-1, so, o, 0.0, 1.0);
    if (m_batchmode)
      l_err = m_net->learnbatch(l_pat+1==patterns());      
    else
      l_err = m_net->learn();
    m_pattern[l_pat].err = l_err;
    l_pat = (l_pat + 1) % patterns();
  }
  else
    l_pat = 0;
  if (m_avg_err==0)
    m_avg_err = l_err;
  else
    m_avg_err = (m_avg_err * c_residual) + l_err * (1 - c_residual);
  return l_err;
}

nnif::PATTERN &nnif::query(int a_patno)
{
  PATTERN l_pat;
  if (m_pattern.find(a_patno)==m_pattern.end()) return m_test;
  m_test = m_pattern[a_patno];
  
  int si, so = m_test.out.size();
  double i[(si = m_test.inp.size())+1];

  for (int n = 0; n < si; n++) i[n] = m_test.inp[n];
  int ivals = 0;
  for (int lay = 0; lay < m_net->rows(); lay++)
  {
    if (ltypeof(lay)==1)
    {
      int nvals = m_net->row(lay)->count();
      m_net->set_input(lay, si-ivals, &i[ivals], 0.0, 1.0);
      ivals += nvals;
    }    
    if (ivals >= si) break;
  }
  
  m_net->docalc();
  
  for (int n = 0; n < so; n++) m_test.out[n] = (*m_net)[n];
  return m_test;
}

void nnif::disp()
{
  ogl_model l_mod;
  int l_rows = m_net->rows();
  for (int n = 0; n < l_rows; n++)
  {
    double y = -1 * ((double)(n+1) / (double)(l_rows+1) - 0.5);
    connection *l_con = m_net->row(n);
    if (!l_con) continue;
    int    l_cols = l_con->count();
    
    for (int m = 0; l_con!=NULL && m < l_cols; m++)
    {
//      double i      = (l_con->isignal()>0.5)?1:0;
      double i      = l_con->isignal();
      double b      = l_con->pbias();
      double co     = l_con->psignal();
      int    l_to   = l_con->tocount();
      
      double x = (double)(l_con->col()+1) / (double)(l_cols + 1) - 0.5;
      
      for (int c = 0; c < l_to; c++)
      {
	double w  = (l_con->toweight(c)-0.5) * 2.0;
	double ac = w * co;
	double cx = l_con->toconn(c)->col();
	double cy = l_con->toconn(c)->row();
	double cb = l_con->toconn(c)->pbias();
	int l_lcount = m_net->row((int)cy)->count();
	
	cx = (cx + 1) / (double)(l_lcount + 1) - 0.5;
	cy = -1 * ((cy + 1) / (double)(l_rows + 1) - 0.5);
	l_mod.linewidth((fabs(w)+0.1)*3);
	l_mod.colour(fabs(w), ac<0?1.0:0.0, ac>0?1.0:0.0, 0.1+fabs(ac)*0.99);
	l_mod.line(x, y-0.015, b, cx, cy+0.015, cb);
      }
      l_mod.end();
      l_mod.flush();
      l_mod.pointsize(fabs(i+2)*3);
      if (i > 0.5)
	l_mod.colour(0.4, fabs(i)*0.5+0.5, 0.4, 0.5+fabs(co)*0.5);
      else
	l_mod.colour(fabs(i)*0.5+0.5, 0.4, 0.9, 0.7+fabs(co)*0.3);
	
      l_mod.point(x,y,b);
      l_mod.end();
      l_mod.flush();
      l_con = l_con->next();
    }
  }
  l_mod.end();
  l_mod.render();
}

void nnif::anavalues(int src, int target, 
		     double from, double through, double step,
		     double start, double len, double skip) {
  m_anavals.src = src;
  m_anavals.target = target;
  m_anavals.from = from;
  m_anavals.through = through;
  m_anavals.step = step;
  m_anavals.start = (long)start;
  m_anavals.len = (long)len;
  m_anavals.skip = (long)skip;
  m_anavals.avg  = 0;
  m_anavals.curr = 0;
  m_anavals.conf = 0;
  m_anavals.avg_ind = 0;
  m_anavals.curr_ind = 0;
  
  ogl_model *l_mod = (ogl_model *)m_anavals.mod;
  if (l_mod) delete l_mod;
  l_mod = new ogl_model();
  m_anavals.mod = (void *)l_mod;
    
  l_mod->linewidth(3);
  l_mod->colour(0.2,0.8,0.8,1.0);    // white
  l_mod->line(0,0,0,1,0,0); l_mod->end();   // x-axis
  l_mod->colour(0.8,0.2,0.8,1.0);    // white
  l_mod->line(0,0,0,0,1,0); l_mod->end();   // y-axis
  l_mod->colour(0.8,0.8,0.2,1.0);  // white
  l_mod->line(0,0,0,0,0,1); l_mod->end();   // z-axis
  l_mod->end();

  l_mod->font("Helvetica-Bold.txf");
//  l_mod->fontrigid(true);
  l_mod->fontsize(0.09);
  l_mod->fontcolor(0.2,0.8,0.8,1.0);    // white
  l_mod->text_align(ogl_model::align_left);
  l_mod->text(-1, -0.94, 0, "Patterns");
  l_mod->fontcolor(0.8,0.8,0.2,1.0);  // white
  l_mod->text_align(ogl_model::align_center);
  l_mod->text(0, -0.94, 0, "Output");
  l_mod->fontcolor(0.8,0.2,0.8,1.0);    // white
  l_mod->text_align(ogl_model::align_right);
  l_mod->text(1, -0.94, 0, "Samples");
  l_mod->end();
  
  l_mod->linewidth(1);
  if (m_anavals.through < m_anavals.from)
  {
    double l_tmp = m_anavals.through;
    m_anavals.through = m_anavals.from;
    m_anavals.from = l_tmp;
  }
  int l_patmin = patterns() - m_anavals.start;
  double l_step = (m_anavals.through - m_anavals.from) / m_anavals.step;
  int l_max_pat = m_anavals.len / m_anavals.skip;
  int l_max_in = (int)m_anavals.step;
  
  float l_gp[l_max_pat + 1][l_max_in + 1];
  float l_er[l_max_pat + 1][l_max_in + 1];
  double l_max_val = 0, l_min_val = 1;
  int m = 0;  l_max_pat = 0; l_max_in = 0;
  for (int l_pat = l_patmin; 
       l_pat < patterns() && l_pat <= l_patmin + m_anavals.len;
       l_pat += m_anavals.skip)
  {
    int    si;
    double i[(si = m_pattern[l_pat].inp.size())+1];
//    double l_lastx = 0;
//    double l_lasty = 0;
//    bool   l_started = false;
//    double l_time = (double)(l_pat-l_patmin) / (double)m_anavals.len;
    for (int n = 0; n < si; n++) i[n] = m_pattern[l_pat].inp[n];
    
    double l_val, l_conf = 0;
    int n = 0;
//    for (l_val = m_anavals.from; 
//	 l_val < m_anavals.through; 
//	 l_val += l_step)
    for (n = 0; n < (int)m_anavals.step; n++)
    {
      l_val = m_anavals.from + n * l_step;
      double l_sval = 
	(l_val - m_imin[m_anavals.src]) / 
	(m_imax[m_anavals.src] - m_imin[m_anavals.src]);
      
      i[m_anavals.src] = l_sval;
      m_net->set_input(0, si, i, 0.0, 1.0);
      m_net->docalc();
      double l_out = (*m_net)[m_anavals.target];
      if (l_out > l_max_val) l_max_val = l_out;
      if (l_out < l_min_val) l_min_val = l_out;
      // m is the pattern choice over time
      // n is the value of the analysis variable
      l_gp[m][n] = l_out;  // value of the grid point
      l_er[m][n] = l_out-m_pattern[l_pat].out[m_anavals.target];
      l_conf += fabs(l_er[m][n]);    // degree of confidence
      if (fabs(l_er[m][n]) > 1.0) l_er[m][n] = 1.0;
      m_anavals.avg  += l_er[m][n];
      if (l_pat + m_anavals.skip >= l_patmin + m_anavals.len) 
	m_anavals.curr += l_er[m][n];
      l_er[m][n] = l_er[m][n] * l_er[m][n];
//      l_er[m][n] = 1.0-1.0/fabs(log10(l_er[m][n]/10.0));
    }
    
    l_max_in = n;
    l_max_pat = ++m;
    l_conf = l_conf / n;
    m_anavals.conf += l_conf;
//    printf("total=%.2f; conf=%.2f\n", m_anavals.conf, l_conf);
  }
  
  m_anavals.conf = m_anavals.conf / m;
  if (m_anavals.conf > 1.0) m_anavals.conf = 1.0;
  if (m_anavals.conf < 0.0) m_anavals.conf = 1.0e-16;
  m_anavals.conf = 1.0 - sqrt(m_anavals.conf);
  m_anavals.curr = m_anavals.curr / l_max_in;
  m_anavals.avg = m_anavals.avg / (l_max_pat * l_max_in);
  
//  if (l_max_val != l_min_val)
//    m_anavals.conf = m_anavals.conf / (l_max_val - l_min_val);
//  m_anavals.conf = 1.0 - (sqrt(m_anavals.conf / (l_max_pat * l_max_in)));
  
  m_anavals.avg_ind = m_anavals.avg>0?1:m_anavals.avg<0?-1:0;
  m_anavals.curr_ind = m_anavals.curr>0?1:m_anavals.curr<0?-1:0;
  m_anavals.avg = fabs(m_anavals.avg);
  m_anavals.curr = fabs(m_anavals.curr);
  
  m_anavals.val = m_pattern[patterns()-1].out[m_anavals.target];
  m_anavals.val *= (m_max[m_anavals.target] - m_min[m_anavals.target]);
  m_anavals.val += m_min[m_anavals.target];
  
  for (int z=1; z < l_max_pat; z++)
  {
    for (int x=1; x < l_max_in; x++)
    {
      double q0 = (double)(x-1)/(l_max_in-1), q1 = (double)(x)/(l_max_in-1);
      double p0 = (double)(z-1)/(l_max_pat-1), p1 = (double)(z)/(l_max_pat-1);
      double l_err, r, g, b;
      double ya, yb, yc, yd;
      
      ya=l_gp[z-1][x-1]; // p0, q0
      yb=l_gp[z-1][x];   // p0, q1
      yc=l_gp[z][x-1];   // p1, q0
      yd=l_gp[z][x];     // p1, q1
      
//      p0 = (l_max_pat + 1.0) / l_max_pat; 
//      p1 *= (l_max_pat + 1.0) / l_max_pat;
//      q0 *= (l_max_in + 1.0) / l_max_in; q1 *= (l_max_in + 1.0) / l_max_in;
      
      l_err = l_er[z-1][x-1] + l_er[z-1][x] + l_er[z][x-1] + l_er[z][x];
      
      l_err = l_err / 4;
      r = l_err;
//      g = (ya+yb+yc+yd)/4;
      g = (1 - l_err);
      b = 0.5;
      
      l_mod->colour(r,g,b, 1 - l_err*0.75);
      l_mod->triangle(p0, ya, q0,
		      p0, yb, q1,
		      p1, yc, q0);
      l_mod->triangle(p1, yc, q0,
		      p1, yd, q1,
		      p0, yb, q1);
    }
    l_mod->end();    
    l_mod->flush();
  }
  l_mod->colour(0.55,0.75,1,1);
  l_mod->linewidth(2);
  for (int x=1; x < l_max_pat; x++)
  {
    int    si, l_pat = l_patmin+(x-1)*m_anavals.skip;
    
    double i[(si = m_pattern[l_pat].inp.size())+1];    
    double p0 = (double)(x-1)/(l_max_pat-1), p1 = (double)(x)/(l_max_pat-1);
    double v0 = m_pattern[l_pat].out[m_anavals.target];
    double v1 = m_pattern[l_pat+m_anavals.skip].out[m_anavals.target];
    
    for (int n = 0; n < si; n++) i[n] = m_pattern[l_pat].inp[n];
    m_net->set_input(0, si, i, 0.0, 1.0);
    m_net->docalc();
    double l_out0 = (*m_net)[m_anavals.target];
    for (int n = 0; n < si; n++) i[n] = m_pattern[l_pat+m_anavals.skip].inp[n];
    m_net->set_input(0, si, i, 0.0, 1.0);
    m_net->docalc();
    double l_out1 = (*m_net)[m_anavals.target];
    l_mod->line(p0,l_out0,v0,p1,l_out1,v1);
    
  }
  l_mod->end();      
  l_mod->render();
};

void nnif::anavalues(int id1, int id2, int target,
		     double from1, double through1, double step1,
		     double from2, double through2, double step2,
		     int patno) {
  m_anavals.target   = target;
  m_anavals.src      = id1;
  m_anavals.from     = from1;
  m_anavals.through  = through1;
  m_anavals.step     = step1;
  m_anavals.src2     = id2;
  m_anavals.from2    = from2;
  m_anavals.through2 = through2;
  m_anavals.step2    = step2;
  m_anavals.start    = patno;

  ogl_model *l_mod = (ogl_model *)m_anavals.mod;
  if (l_mod) delete l_mod;
  l_mod = new ogl_model();
  m_anavals.mod = (void *)l_mod;

  l_mod->linewidth(3);
  l_mod->colour(0.2,0.8,0.8,1.0);    // white
  l_mod->line(0,0,0,1,0,0); l_mod->end();   // x-axis
  l_mod->colour(0.8,0.2,0.8,1.0);    // white
  l_mod->line(0,0,0,0,1,0); l_mod->end();   // y-axis
  l_mod->colour(0.8,0.8,0.2,1.0);  // white
  l_mod->line(0,0,0,0,0,1); l_mod->end();   // z-axis
  l_mod->end();
  
  int l_pat = patterns() - m_anavals.start - 1;
  if (l_pat < 0) l_pat = 0;
  
  double l_step1 = (m_anavals.through - m_anavals.from) / m_anavals.step;  
  double l_step2 = (m_anavals.through2 - m_anavals.from2) / m_anavals.step2;  

  float l_gp[(int)m_anavals.step + 1][(int)m_anavals.step2 + 1];

  double l_max_val = 0, l_min_val = 1;
  for (int z = 0; z < m_anavals.step; z++)
  {
    int si;
    double i[(si = m_pattern[l_pat].inp.size())+1];
    for (int n = 0; n < si; n++) i[n] = m_pattern[l_pat].inp[n];
    
    double l_val = m_anavals.from + z * l_step1;
    double l_sval = 
      (l_val - m_imin[m_anavals.src]) / 
      (m_imax[m_anavals.src] - m_imin[m_anavals.src]);
    
    i[m_anavals.src] = l_sval;
    
    for (int x = 0; x < m_anavals.step2; x++)
    {
      l_val = m_anavals.from + x * l_step2;
      l_sval = 
	(l_val - m_imin[m_anavals.src2]) / 
	(m_imax[m_anavals.src2] - m_imin[m_anavals.src2]);
      
      i[m_anavals.src2] = l_sval;
      
      m_net->set_input(0, si, i, 0.0, 1.0);
          
      m_net->docalc();
      double l_out = (*m_net)[m_anavals.target];
      
      l_out = l_out - m_pattern[l_pat].out[m_anavals.target];
      l_out *= l_out;
      
      if (l_out > l_max_val) l_max_val = l_out;
      if (l_out < l_min_val) l_min_val = l_out;
      l_gp[z][x] = l_out;
    }  
  }
  
  for (int z=1; z < m_anavals.step; z++)
  {
    for (int x=1; x < m_anavals.step2; x++)
    {
      double q0 = (double)(x-1)/(m_anavals.step2-1);
      double q1 = (double)(x)/(m_anavals.step2-1);
      double p0 = (double)(z-1)/(m_anavals.step-1);
      double p1 = (double)(z)/(m_anavals.step-1);
      double l_err, r, g, b;
      double ya, yb, yc, yd;
      
      ya=l_gp[z-1][x-1]; // p0, q0
      yb=l_gp[z-1][x];   // p0, q1
      yc=l_gp[z][x-1];   // p1, q0
      yd=l_gp[z][x];     // p1, q1
      
      ya = (ya - l_min_val) / (l_max_val - l_min_val);
      yb = (yb - l_min_val) / (l_max_val - l_min_val);
      yc = (yc - l_min_val) / (l_max_val - l_min_val);
      yd = (yd - l_min_val) / (l_max_val - l_min_val);
      
//      printf("%.2f; %.2f; %.2f; %.2f\n", ya, yb, yc, yd);
      l_err = (ya+yb+yc+yd)/4.0;
      l_err = 1 - l_err;
      r = l_err;
      g = r*0.95;
      b = 1.0;
      
      l_mod->colour(r,g,b, 0.35+l_err*0.65);
      l_mod->triangle(p0, ya, q0,
		      p0, yb, q1,
		      p1, yc, q0);
      l_mod->triangle(p1, yc, q0,
		      p1, yd, q1,
		      p0, yb, q1);
    }
    l_mod->end();    
    l_mod->flush();
  }
  
  l_mod->linewidth(1);
}


void  nnif::analyse() 
{
  ogl_model *l_mod = (ogl_model *)m_anavals.mod;
  if (l_mod) l_mod->render();
}

void  nnif::load(const char *a_fname, SAVE_TYPE a_fmt)
{
  char *l_ext;
  string l_fname;
  if (a_fname==NULL) return;
  switch (a_fmt)
  {
   case save_arch     : l_ext = ".net"; break;
   case save_model    : l_ext = ".mod"; break;
   case save_patterns : l_ext = ".pat"; break;
   default : l_ext = "";
  }
  if (strstr(a_fname, l_ext)==NULL)
    l_fname = string(a_fname) + string(l_ext);
  else
    l_fname = string(a_fname);
    
//  printf("Loading %s\n", l_fname.c_str());
  
  FILE *l_file = fopen(l_fname.c_str(), "r");
  if (l_file)
  {
    char l_buf[4096];
    if (a_fmt==save_model || a_fmt==save_arch) 
    {
      m_pattern.clear();
      reset();
    }
    else
      m_pattern.clear();
    
    while (fgets(l_buf, sizeof(l_buf), l_file))
    {
      int n;
      char *l_fv, *l_pt = l_buf, *e = strchr(l_buf, '\n');
      if (e) *e = 0;
//      printf("%s\n", l_buf);
      if      (strstr(l_buf, ".net:")==l_buf)
      {
//	printf("Network %s\n", l_buf);
      }
      else if (strstr(l_buf, ".pat:")==l_buf)
      {
//	printf("Pattern %s\n", l_buf);
      }
      else if (strstr(l_buf, ".mod:")==l_buf)
      {
//	printf("Model %s\n", l_buf);
      }
      else if (strstr(l_buf, "model:")==l_buf)
      {
	long l_ofs = ftell(l_file);
	fclose(l_file);
	fprintf(stderr, "Reading Model: %s\n", m_params.c_str());
	delete m_net;
	m_net = new network(paramlist(m_params.c_str()));
	ifstream fin(l_fname.c_str());
	fin.seekg(l_ofs);
	fin >> *m_net;
	l_ofs = fin.tellg();
	l_file = fopen(l_fname.c_str(), "r");
	fseek(l_file, l_ofs, SEEK_SET);
      }
      else if (strstr(l_buf, "parm:")==l_buf)
      {
	params() = string(&l_buf[5]);
      }
      else if (strstr(l_buf, "Row\t")==l_buf)
      {
	l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	if (l_fv) 
	{
	  n = atoi(l_fv);
	  m_layer[n].type = 0;
	  m_layer[n].text = "";
	  m_layer[n].ncount = 0;
	  
	  if (l_pt) 
	  {	    
	    l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	    m_layer[n].text = string(l_fv);
	  }
	  if (l_pt)
	  {	    
	    l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	    m_layer[n].ncount = atoi(l_fv);
	  }
	  if (l_pt)
	  {	    
	    l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	    m_layer[n].type   = atoi(l_fv);
	  }	  
	}	
      }
      else if (strstr(l_buf, "Conn\t")==l_buf)
      {
	CONNPAIR c;
	l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	if (l_fv) c.first.layer = atoi(l_fv);
	l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	if (l_fv) c.first.node = atoi(l_fv);
	l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	if (l_fv) c.second.layer = atoi(l_fv);
	l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	if (l_fv) c.second.node = atoi(l_fv);
	m_connect.push_back(c);
      }
      else if (strstr(l_buf, "lab_in\t")==l_buf)
      {
	l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	n = 0;
	while (l_pt)
	{
	  l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	  m_ilab[n++] = string(l_fv);
	}
      }
      else if (strstr(l_buf, "lab_out\t")==l_buf)
      {
	l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	n = 0;
	while (l_pt)
	{
	  l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	  m_olab[n++] = string(l_fv);
	}
      }
      else if (strstr(l_buf, "in_minmax\t")==l_buf)
      {
	l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	n = atoi(l_fv);
	l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	double l_min = atof(l_fv);
	l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	double l_max = atof(l_fv);
	m_imin[n] = l_min;
	m_imax[n] = l_max;	
      }
      else if (strstr(l_buf, "out_minmax\t")==l_buf)
      {
	l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	n = atoi(l_fv);
	l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	double l_min = atof(l_fv);
	l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	double l_max = atof(l_fv);
	m_min[n] = l_min;
	m_max[n] = l_max;	
      }
      else if (strstr(l_buf, "pat_in\t")==l_buf)
      {
	l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	n = atoi(l_fv);
	l_fv = l_pt; if (l_fv) l_pt = strchr(l_fv, '\t'); 
	if (l_pt) *l_pt++ = 0;
	int i = 0;
	while (l_fv)
	{
	  m_pattern[n].inp[i++] = atof(l_fv);
	  l_fv = l_pt; if (l_fv) l_pt = strchr(l_fv, '\t'); 
	  if (l_pt) *l_pt++ = 0;
	}	
      }
      else if (strstr(l_buf, "pat_out\t")==l_buf)
      {
	l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	l_fv = l_pt; l_pt = strchr(l_fv, '\t'); if (l_pt) *l_pt++ = 0;
	n = atoi(l_fv);
	l_fv = l_pt; if (l_fv) l_pt = strchr(l_fv, '\t'); 
	if (l_pt) *l_pt++ = 0;
	int i = 0;
	while (l_fv)
	{
	  m_pattern[n].err = 0;
	  m_pattern[n].out[i++] = atof(l_fv);
	  l_fv = l_pt; if (l_fv) l_pt = strchr(l_fv, '\t'); 
	  if (l_pt) *l_pt++ = 0;
	}	
      }
    }    
    fclose(l_file);
  }
  if (a_fmt==save_arch) build();
}

void  nnif::save(const char *a_fname, SAVE_TYPE a_fmt)
{
  char *l_ext;
  string l_fname;
  if (a_fname==NULL) return;
  switch (a_fmt)
  {
   case save_arch     : l_ext = ".net"; break;
   case save_model    : l_ext = ".mod"; break;
   case save_patterns : l_ext = ".pat"; break;
   default : l_ext = "";
  }
  if (strstr(a_fname, l_ext)==NULL)
    l_fname = string(a_fname) + string(l_ext);
  else
    l_fname = string(a_fname);
    
  fprintf(stderr, "Saving %s\n", l_fname.c_str());
  
  FILE *l_file = fopen(l_fname.c_str(), "w");
  if (l_file)
  {
    fprintf(l_file, "%s:in=%i:out=%i\n", 
	    l_ext, no_inputs(), no_outputs());
    if (a_fmt==save_model || a_fmt==save_arch)
    {
      fprintf(l_file, "parm:%s\n", params().c_str());
      map <int, LAYER>::iterator layer;
      for (layer=m_layer.begin(); layer!=m_layer.end(); ++layer)
      {
	if (layer->second.ncount>0)
	{
	  fprintf(l_file, "Row\t%i\t%s\t%i\t%i\n", 
		  layer->first,
		layer->second.text.c_str(), 
		  layer->second.ncount,      
		  layer->second.type);
	}
      }
      
      deque <CONNPAIR>::iterator c;
      for (c=m_connect.begin(); c != m_connect.end(); ++c)
      {
	fprintf(l_file, "Conn\t%i\t%i\t%i\t%i\n",
		c->first.layer, c->first.node, 
		c->second.layer, c->second.node);
      }
    }
    
    if (m_ilab.size())
    {
      fprintf(l_file, "lab_in");
      for (int n = 0; (unsigned)n < m_ilab.size(); n++)
	fprintf(l_file, "\t%s", m_ilab[n].c_str());
      fprintf(l_file, "\n");
    }
    
    if (m_olab.size())
    {
      fprintf(l_file, "lab_out");
      for (int n = 0; (unsigned)n < m_olab.size(); n++)
	fprintf(l_file, "\t%s", m_olab[n].c_str());
      fprintf(l_file, "\n");
    }
    
    for (int n = 0; n < no_inputs(); n++)
    {
      if (m_imin.find(n)==m_imin.end()) m_imin[n] = 0;
      if (m_imax.find(n)==m_imax.end()) m_imax[n] = 1;
      fprintf(l_file, "in_minmax\t%i\t%.12f\t%.12f\n", 
	      n, m_imin[n], m_imax[n]);      
    }
    
    for (int n = 0; n < no_outputs(); n++)
    {
      if (m_min.find(n)==m_min.end()) m_min[n] = 0;
      if (m_max.find(n)==m_max.end()) m_max[n] = 1;
      fprintf(l_file, "out_minmax\t%i\t%.12f\t%.12f\n", 
	      n, m_min[n], m_max[n]);      
    }
    
    map <int, PATTERN>::iterator p;
    for(p=m_pattern.begin(); p!=m_pattern.end(); ++p)
    {
      NVALUE::iterator n;
      fprintf(l_file, "pat_in\t%i", p->first);
      for (n=p->second.inp.begin(); n != p->second.inp.end(); ++n)
	fprintf(l_file, "\t%.4f", n->second);
      fprintf(l_file, "\npat_out\t%i", p->first);
      for (n=p->second.out.begin(); n != p->second.out.end(); ++n)
	fprintf(l_file, "\t%.4f", n->second);
      fprintf(l_file, "\n");
    }
    if (a_fmt==save_model)
    {
      fprintf(l_file, "model:\n");
      fclose(l_file);
      ofstream fout(l_fname.c_str(), ofstream::app);
      fout << *m_net;
    }    
    else
      fclose(l_file);
  }
}

