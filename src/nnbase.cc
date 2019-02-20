/*=========================================================================*\
\*=========================================================================*/
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <typeinfo>
#include "nnbase.h"

int g_nconns = 0;
/*=======================================================================*/
inline double def_xfer(double a_val)
{
  return 1.0/(1.0 + exp(-(a_val)));
}

/*=======================================================================*/
inline double def_getval(double a_val)
{
  return log(1.0/a_val-1.0)*-1;
}

/*=======================================================================*/
inline double def_diff(double a_val)
{
//  return 0.1 + 2 * a_val * (1.0 - a_val);
  return a_val * (1.0 - a_val);
}


/*=======================================================================*/
void paramlist::readparm(int n, const char *a_parm) 
{
  if (a_parm==NULL) return;
  if (a_parm[0]==0) return;
  char *t = strdup(a_parm);
  pv[n] = t + strcspn(t, "=:\t");
  *pv[n]++ = 0;
  pv[n] = pv[n] + strspn(pv[n], "=:\t");
  pv[n] = strdup(pv[n]);
  p[n]  = strdup(t);
  free(t);
  np++;
}

/*=======================================================================*/
paramlist::paramlist(const char *a_params) 
{
  char *p[MAX_PARAM+1], *l_pt = strdup(a_params), *l_tmp = l_pt;
  int n = 0;
  fprintf(stderr, "Paramlist=[%s]\n", a_params);
  p[0] = NULL;
  while (l_pt && l_pt[0])
  {
    p[n++] = l_pt; l_pt += strcspn(l_pt, " \t"); p[n] = NULL;
    if (l_pt[0]!=0) *l_pt++ = 0;
  }
  
  np = 0;
  for (n = 0; n < MAX_PARAM && p[n]!=NULL; n++) readparm(n, p[n]);  
  
  free(l_tmp);
}

/*=======================================================================*/
paramlist::paramlist(char *params[]) 
{ 
  np = 0;
  for (int n = 0; n < MAX_PARAM && params[n]!=NULL; n++)
    readparm(n, params[n]);
};
/*=======================================================================*/
paramlist::paramlist(char *p1, char *p2, char *p3, char *p4,
		     char *p5, char *p6, char *p7, char *p8, 
		     char *p9, char *p10, char *pmax) 
{ 
  np = 0;
  if (p1!=NULL)  readparm(0, p1);
  if (p2!=NULL)  readparm(1, p2);
  if (p3!=NULL)  readparm(2, p3);
  if (p4!=NULL)  readparm(3, p4);
  if (p5!=NULL)  readparm(4, p5);
  if (p6!=NULL)  readparm(5, p6);
  if (p7!=NULL)  readparm(6, p7);
  if (p8!=NULL)  readparm(7, p8);
  if (p9!=NULL)  readparm(8, p9);
  if (p10!=NULL) readparm(9, p10);
};
   
/*=======================================================================*/
paramlist::~paramlist() 
{
  for (int n = 0; n < np; n++) {
    free(pv[n]);
    free(p[n]);
  }
};
   
/*=======================================================================*/
paramlist &paramlist::operator=(const paramlist &c) 
{ 
  np = 0;
  for (int n = 0; n < MAX_PARAM && c[n]; n++)
    readparm(n, c[n]);
  return *this;
};
   
/*=======================================================================*/
const char *paramlist::operator[](const char *s) const 
{
  for (int n = 0; n < np; n++)
    if (!strcasecmp(s, p[n])) return pv[n];
  return "";
}

/*=======================================================================*/
const char *paramlist::operator[](const int i) const 
{ 
  static char b[1024];
  if (i >= np) return "";
  sprintf(b, "%s=%s", p[i], pv[i]);
  return b;
};
   
/*=======================================================================*/
const char *paramlist::getparm(const char *a_parm, 
			       const char *a_default) const 
{
  for (int n = 0; n < np; n++)
    if (!strcasecmp(a_parm, p[n])) return pv[n];
  return a_default;
}

istream& paramlist::get(istream &i) 
{
  char l_buf[256];
  
  for (int n = 0; n < np; n++) {
    free(pv[n]);
    free(p[n]);
  }
  np = 0;
  
  for (int n = 0; n < MAX_PARAM; n++) {
    i.getline(l_buf, sizeof(l_buf));
    if (!strcmp(l_buf, "endparam")) break;
    readparm(n, l_buf);
  }
  return i;
}

ostream& paramlist::put(ostream &o) const 
{
  char l_buf[256];
  for (int n = 0; n < np; n++) {
    sprintf(l_buf, "%s=%s\n", p[n], pv[n]);
    o.write(l_buf, strlen(l_buf));
  }
  o << "endparam\n";
  return o;
}

/*=======================================================================*/
/*=======================================================================*/
connection::neurode::neurode(connection *a_from, transfer *a_trans)
{
  xfer      = 0;
  bias      = 0;
  eta       = 0.75;
  lambda    = 0.001;
  trans     = a_trans;
  tolist    = NULL;
  fromcon   = a_from;        // single 'connection object'
  tocount   = 0;
  weights   = NULL;
  cweight   = NULL;
}

/*=======================================================================*/
connection::neurode::neurode(connection *a_from, 
			     connection *a_to, 
			     transfer   *a_trans)
{
  xfer      = 0;
  bias      = 0;
  lambda    = 0.01;          // default momentum constant
  eta       = 0.75;          // default node learning rate
  trans     = a_trans;
  fromcon   = a_from;          // single 'connection object'
  tolist  = NULL;
  tocount = 0;
  weights  = NULL;
  cweight  = NULL;
  connect_to_layer(a_to);
}

/*=======================================================================*/
connection::neurode::~neurode()
{
  if (tolist)   free(tolist);   tolist = NULL;
  if (weights)  free(weights);  weights = NULL;
  if (cweight)  free(cweight);  cweight = NULL;
}

/*=======================================================================*/
void connection::neurode::connect(connection *a_target)
{
  if (!a_target) return;
  printf("connect %s to %s\n", fromcon->tag(), a_target->tag());
  
  tocount++;
  tolist = (connection **)realloc(tolist, tocount * sizeof(connection *));
  tolist[tocount-1] = a_target;
  
  weights   = (weight *)realloc(weights,  tocount * sizeof(weight));
  cweight   = (weight *)realloc(cweight,  tocount * sizeof(weight));
  
  weights[tocount-1].momentum() = 0;
  weights[tocount-1]   =  weight(1.0-((rand() * 2.0/(double)RAND_MAX)));
  cweight[tocount-1]   =  weight(1.0-((rand() * 2.0/(double)RAND_MAX)));
//  fprintf(stderr, "tocount=%i; weight=%.4g; cweight=%.4g\n", 
//	  tocount, (double)weights[tocount-1], (double)cweight[tocount-1]);
}

/*=======================================================================*/
void connection::neurode::connect_to_layer(connection *a_out)
{
  if (!a_out) return;
//  printf("connect %s to %s\n", fromcon->tag(), a_out->tag());
  connect(a_out);  // make double connections for counterweights
//  if (typeid(*this)==typeid(backprop_cw)) connect(a_out);
  connect_to_layer(a_out->next());
}

/*=======================================================================*/
void connection::neurode::fire()
{
  if (fromcon->is_input()) fromcon->input();
  if (fromcon->ignore())
    xfer = 0;
  else
    xfer = trans->xfer(fromcon->total() + bias);
  
  for (int n=0; n<tocount; n++)
    tolist[n]->update(xfer * (weights[n] + cweight[n]));
  
  fromcon->do_output(bias);
}

/*=======================================================================*/
istream &connection::delta_bar_delta::get_info(GRID g, istream &i)
{
  char l_buf[256];
  i >> tocount >> bias >> eta >> lambda ;
  i >> theta >> kappa >> decay >> max_eta ;
  i.getline(l_buf, sizeof(l_buf));
  tolist = (connection **)realloc(tolist, tocount * sizeof(connection *));
  weights   = (weight *)realloc(weights,  tocount * sizeof(weight));
  cweight   = (weight *)realloc(cweight,  tocount * sizeof(weight));
  for (int n = 0; n < tocount; n++)
  {
    i.getline(l_buf, sizeof(l_buf));
    if (strcmp(l_buf, "<to>")) 
    {
      fprintf(stderr, "dbd - (mismatch) expected <to>, got %s\n", l_buf);
      return i;
    }
    i.getline(l_buf, sizeof(l_buf));   // targeted tag
    tolist[n] = NULL;
    for (int p=0; g[p]!=NULL; p++)
      if ((tolist[n] = g[p]->find(l_buf))!=NULL)
	break;
    if (tolist[n]==NULL) 
      fprintf(stderr, "%s: target '%s' not found\n", 
	      fromcon->tag(), l_buf);
    i >> weights[n];
    i >> cweight[n];
    i >> weights[n].momentum();
    i >> weights[n].eta();
    i >> weights[n].slope();
    i.getline(l_buf, sizeof(l_buf));
    i.getline(l_buf, sizeof(l_buf)); // </to>
//    if (fabs(weights[n]) < 1.0e-3) 
//    {
//      printf("pruning %s->%s\n", fromcon->tag(), tolist[n]->tag());
//      tocount--;
//      n--;
//    }
//    else
      g_nconns++;
  }
  if (tocount==0 && !fromcon->is_input() && !fromcon->is_output())
  {
    fprintf(stderr, "%s has no targets left: deactivating!\n", fromcon->tag());
    fromcon->set_unused();
  }
  i.getline(l_buf, sizeof(l_buf));   // </delta-bar-delta>
  return i;
}

/*=======================================================================*/
ostream &connection::delta_bar_delta::put_info(ostream &o)
{
  for (int n = 0; n < tocount; n++)
    if (fabs(weights[n] + cweight[n]) < 1.0e-3)
    {
      fprintf(stderr, "pruning %s->%s\n", fromcon->tag(), tolist[n]->tag());
      memmove(&tolist[n], &tolist[n+1], (tocount-n-1) * sizeof(connection *) );
      memmove(&weights[n], &weights[n+1], (tocount-n-1) * sizeof(weight) );
      memmove(&cweight[n], &cweight[n+1], (tocount-n-1) * sizeof(weight) );
      tocount--;
      n--;
    }
  if (tocount==0 && !fromcon->is_input() && !fromcon->is_output())
  {
    fprintf(stderr, "%s has no targets left: deactivating!\n", fromcon->tag());
    fromcon->set_unused();
    return o;
  }
  
  o << "<delta-bar-delta>\n";
  o << tocount <<'\t'<< bias <<'\t'<< eta << '\t' << lambda << '\t';
  o << theta << '\t' << kappa << '\t' << decay << '\t' << max_eta <<'\n';
  for (int n = 0; n < tocount; n++)
  {
    o << "<to>\n";
    o << tolist[n]->tag() << '\n';
    o << weights[n] << '\t' << cweight[n] << '\t';
    o << weights[n].momentum() << '\t';
    o << weights[n].eta() << '\t' << weights[n].slope() << '\n';
    o << "</to>\n";
  }
  o << "</delta-bar-delta>\n";
  return o;
}

/*=======================================================================*/
void connection::delta_bar_delta::adjust(bool a_set_err, bool a_epoch)
{
  double l_err, l_delta, l_sigma = 0, l_chng;
  if (fromcon->is_output()) 
  {
    l_sigma = trans->xfer(fromcon->fixed()) - xfer;
//    printf("Output %s: sigma=%.4g; xfer=%.4g\n", fromcon->tag(), l_sigma, xfer);
  }
  else for (int n=0; n<tocount; n++)
  {
    l_err = tolist[n]->error();
    if (fabs(l_err) < 1.0e-120) l_err = 0;
    l_sigma += l_err * weights[n];
    
    l_err *= xfer;
    weights[n].slope() = (1-theta)*l_err + theta*weights[n].slope();

    if (a_epoch)
    {
      l_chng  = weights[n].slope() * l_err;
      if      (l_chng < 0)   // sign is different
	weights[n].eta() *= (1-decay);
      else if (l_chng > 0)   // sign is the same
	weights[n].eta() += kappa;
      
      if (weights[n].eta() > max_eta)  weights[n].eta() = max_eta;
      
      l_delta  =  l_err * weights[n].eta() - weights[n] * decay;
      weights[n] += weights[n].momentum() + l_delta;
      weights[n].momentum() = l_delta * lambda;
    }    
  }
  if (a_set_err) 
  {
    bias += eta * l_sigma * trans->diff(bias+0.5);
    l_sigma = trans->diff(xfer) * l_sigma;
    fromcon->set_error(l_sigma);
  }
  else
  {
    l_sigma = trans->diff(xfer) * l_sigma;    
    fromcon->set_error(fromcon->error()+l_sigma);
  }  
}

/*=======================================================================*/
istream &connection::backprop_decay::get_info(GRID g, istream &i)
{
  char l_buf[256];
  i >> tocount;
  i >> bias;
  i >> eta;
  i >> lambda ;
  i >> decay ;
  i.getline(l_buf, sizeof(l_buf));
  tolist = (connection **)realloc(tolist, tocount * sizeof(connection *));
  weights   = (weight *)realloc(weights,  tocount * sizeof(weight));
  cweight   = (weight *)realloc(cweight,  tocount * sizeof(weight));
  for (int n = 0; n < tocount; n++)
  {
    i.getline(l_buf, sizeof(l_buf));
    if (strcmp(l_buf, "<to>")) 
    {
      fprintf(stderr, "bp_decay - (mismatch) expected <to>, got %s\n", l_buf);
      return i;
    }
    i.getline(l_buf, sizeof(l_buf));   // targeted tag
    tolist[n] = NULL;
    for (int p=0; g[p]!=NULL; p++)
      if ((tolist[n] = g[p]->find(l_buf))!=NULL)
	break;
    if (tolist[n]==NULL) 
      fprintf(stderr, "%s: target '%s' not found\n", 
	      fromcon->tag(), l_buf);
    i >> weights[n];
    i >> cweight[n];
    i >> weights[n].momentum();
    i.getline(l_buf, sizeof(l_buf));
    i.getline(l_buf, sizeof(l_buf)); // </to>
//    if (fabs(weights[n]) < 1.0e-3) 
//    {
//      printf("pruning %s->%s\n", fromcon->tag(), tolist[n]->tag());
//      tocount--;
//      n--;
//    }
//    else
      g_nconns++;
  }
  if (tocount==0 && !fromcon->is_input() && !fromcon->is_output())
  {
    fprintf(stderr, "%s has no targets left: deactivating!\n", fromcon->tag());
    fromcon->set_unused();
  }
  i.getline(l_buf, sizeof(l_buf));   // </backprop_decay>
  return i;
}

/*=======================================================================*/
ostream &connection::backprop_decay::put_info(ostream &o)
{
  o << "<backprop-decay>\n";
  o << tocount << '\t' << bias << '\t' << eta << '\t';
  o << lambda << '\t' << decay << '\n';
  for (int n = 0; n < tocount; n++)
  {
    o << "<to>\n";
    o << tolist[n]->tag() << '\n';
    o << weights[n] << '\t' << cweight[n] << '\t';
    o << weights[n].momentum() << '\n';
    o << "</to>\n";
  }
  o << "</backprop-decay>\n";
  return o;
}
/*=======================================================================*/
void connection::backprop_decay::adjust(bool a_set_err, bool a_epoch)
{
  double l_err, l_delta, l_sigma = 0;
  if (fromcon->is_output()) 
    l_sigma = trans->xfer(fromcon->fixed()) - xfer;
  else for (int n=0; n<tocount; n++)
  {
    l_err = tolist[n]->error();
    if (fabs(l_err) < 1.0e-120) l_err = 0;
    
    l_sigma += l_err * weights[n];    
    if (a_epoch)
    {
      l_delta  = l_err * xfer * eta - weights[n] * decay;
    
      weights[n] += weights[n].momentum() + l_delta;
      weights[n].momentum() = l_delta * lambda;
    }
  }
  l_sigma = trans->diff(xfer) * l_sigma;
  if (a_set_err) 
  {
    bias += eta * l_sigma * trans->diff(bias+0.5);
    fromcon->set_error(l_sigma);
  }
  else
    fromcon->set_error(fromcon->error()+l_sigma);    
}

/*=======================================================================*/
istream &connection::backprop_cw::get_info(GRID g, istream &i)
{
  double l_version = 0.0;
  char l_buf[256];
  i >> tocount >> bias >> eta >> theta >> decay >> lambda ;
  i.getline(l_buf, sizeof(l_buf));
  tolist = (connection **)realloc(tolist, tocount * sizeof(connection *));
  weights   = (weight *)realloc(weights,  tocount * sizeof(weight));
  cweight   = (weight *)realloc(cweight,  tocount * sizeof(weight));
  for (int n = 0; n < tocount; n++)
  {
    i.getline(l_buf, sizeof(l_buf));
    if (strcmp(l_buf, "<to-v1>") && strcmp(l_buf, "<to>")) 
    {
      fprintf(stderr, "backprop_cw - (mismatch) expected <to>, got %s\n", l_buf);
      return i;
    }
    if (!strcmp(l_buf, "<to-v1>")) l_version = 1.0;
    i.getline(l_buf, sizeof(l_buf));   // targeted tag
    tolist[n] = NULL;
    for (int p=0; g[p]!=NULL; p++)
      if ((tolist[n] = g[p]->find(l_buf))!=NULL)
	break;
    if (tolist[n]==NULL) 
      fprintf(stderr, "%s: target '%s' not found\n", 
	      fromcon->tag(), l_buf);
    
    i >> weights[n];
    i >> cweight[n];
    i >> weights[n].momentum();
    
    if (l_version==1.0)
    {
      i >> cweight[n].momentum();
      i >> weights[n].slope();
      i >> cweight[n].slope();
      i >> weights[n].eta();
    }
    else
    {
      cweight[n].momentum() = weights[n].momentum();
      weights[n].slope() = 0.0;
      cweight[n].slope() = 0.0;
      weights[n].eta() = eta;
    }
    i.getline(l_buf, sizeof(l_buf));
    i.getline(l_buf, sizeof(l_buf)); // </to-v1>
    
    g_nconns++;
  }
  if (tocount==0 && !fromcon->is_input() && !fromcon->is_output())
  {
    fprintf(stderr, "%s has no targets left: deactivating!\n", fromcon->tag());
    fromcon->set_unused();
  }
  i.getline(l_buf, sizeof(l_buf));   // </backprop_cw>
  return i;
}

/*=======================================================================*/
ostream &connection::backprop_cw::put_info(ostream &o)
{
  o << "<backprop_cw>\n";
  o << tocount << '\t' << bias << '\t' << eta << '\t';
  o << theta << '\t' << decay << '\t' << lambda << '\n';
  for (int n = 0; n < tocount; n++)
  {
    o << "<to-v1>\n";
    o << tolist[n]->tag() << '\n';
    o << weights[n] << '\t' << cweight[n] << '\t';
    o << weights[n].momentum() << '\t' << cweight[n].momentum() << '\t';
    o << weights[n].slope() << '\t' << cweight[n].slope() << '\t';
    o << weights[n].eta() << '\n';
    o << "</to-v1>\n";
  }
  o << "</backprop_cw>\n";
  return o;
}

/*=======================================================================*/
void connection::backprop_cw::fire()
{
  if (fromcon->is_input()) fromcon->input();
  if (fromcon->ignore())
    xfer = 0;
  else
    xfer = trans->xfer(fromcon->total() + bias);
  
  for (int n=0; n<tocount; n++)
    tolist[n]->update(xfer * (weights[n] + cweight[n]));
  
  fromcon->do_output(bias);
}

double gaussean(double a) { return exp(-(a*a)); }

/*=======================================================================*/
//    printf("%s->%s: (eta=%.5g); diff=%.5g; %.5g <> %.5g\n", 
//	   fromcon->tag(), tolist[n]->tag(), eta, l_diff,
//	   (double)weights[n], (double)cweight[n]);
void connection::backprop_cw::adjust(bool a_set_err, bool a_epoch)
{
  double l_etotal=0, l_err, l_delta, l_sigma=0, l_mid, l_rate=0;
  double l_pe = gaussean(bias);
  if (fromcon->is_output()) 
    l_sigma = trans->xfer(fromcon->fixed()) - xfer;
  else
  {
    if (eta <= 0) eta = 0.001;  // sanity check
    for (int n=0; n<tocount; n++)
    {
      // for weights close to each other, xfer(d[wt]-n) is small, therefore
      // l_rate is large.  Sigma therefore has a larger effect.  When l_rate
      // approaches 1, then sum(1-l_rate) approaches 0.  
      // l_arate=1-sum(1-l_rate) will therefore approach 1 
      // where l_rate approaches 1.
      double w = weights[n], c = cweight[n], a = fabs(w  - c);      
      l_err = tolist[n]->error();
      l_rate = eta + max_eta * gaussean(w-c);
//      l_rate += l_pe;
      double l_out_err = l_err * xfer;
      l_delta = l_out_err * l_rate;
      l_mid = (w + c + l_delta) / 2.0;      
      l_etotal += l_delta;
      
      bool l_counter = fabs(w - l_mid) > fabs(c - l_mid);
      if (fabs(l_out_err) > a) 
        l_counter = !l_counter;
       
      if (l_counter)
      {
	if (!a_epoch) continue;
	weights[n] += weights[n].momentum() + l_delta;
	weights[n].momentum() = l_delta * lambda;
      }
      else
      {
	if (!a_epoch) continue;	
	cweight[n] += cweight[n].momentum() + l_delta;
	cweight[n].momentum() = l_delta * lambda;
      }
      l_sigma += l_err * (w + c);
    }
  }
  l_sigma = trans->diff(xfer) * l_sigma;    
//  if (fabs(l_etotal) > 1) l_etotal /= fabs(l_etotal);
  if (a_set_err && a_epoch) 
  {
//    bias += l_etotal * (1 - l_arate);
    bias *= (1 - decay * l_sigma * l_sigma);
    bias += l_etotal * l_pe * eta;
    fromcon->set_error(l_sigma);
  }
  else
    fromcon->set_error(fromcon->error()+l_sigma);
}

/*=======================================================================*/
istream &connection::backprop::get_info(GRID g, istream &i)
{
  char l_buf[256];
  i >> tocount;
  i >> bias;
  i >> eta;
  i >> lambda ;
  i.getline(l_buf, sizeof(l_buf));
  tolist = (connection **)realloc(tolist, tocount * sizeof(connection *));
  weights   = (weight *)realloc(weights,  tocount * sizeof(weight));
  cweight   = (weight *)realloc(cweight,  tocount * sizeof(weight));
  for (int n = 0; n < tocount; n++)
  {
    i.getline(l_buf, sizeof(l_buf));
    if (strcmp(l_buf, "<to>")) 
    {
      fprintf(stderr, "backprop - (mismatch) expected <to>, got %s\n", l_buf);
      return i;
    }
    i.getline(l_buf, sizeof(l_buf));   // targeted tag
//    printf("Finding [%s]\n", l_buf);
    tolist[n] = NULL;
    for (int p=0; g[p]!=NULL; p++)
      if ((tolist[n] = g[p]->find(l_buf))!=NULL)
	break;
    if (tolist[n]==NULL) 
      fprintf(stderr, "%s: target '%s' not found\n", 
	      fromcon->tag(), l_buf);
//    printf("Connecting [%s] to [%s(%s)]\n", 
//	   fromcon->tag(), l_buf, tolist[n]->tag());
    i >> weights[n];
    i >> cweight[n];
    i >> weights[n].momentum();
    i.getline(l_buf, sizeof(l_buf));
    i.getline(l_buf, sizeof(l_buf)); // </to>
//    if (fabs(weights[n]) < 1.0e-3) 
//    {
//      printf("pruning %s->%s\n", fromcon->tag(), tolist[n]->tag());
//      tocount--;
//      n--;
//    }
//    else
      g_nconns++;
  }
  if (tocount==0 && !fromcon->is_input() && !fromcon->is_output())
  {
    fprintf(stderr, "%s has no targets left: deactivating!\n", fromcon->tag());
    fromcon->set_unused();
  }
  i.getline(l_buf, sizeof(l_buf));   // </backprop>
  return i;
}

/*=======================================================================*/
ostream &connection::backprop::put_info(ostream &o)
{
  o << "<backprop>\n";
  o << tocount << '\t' << bias << '\t' << eta << '\t' << lambda << '\n';
  for (int n = 0; n < tocount; n++)
  {
    o << "<to>\n";
    o << tolist[n]->tag() << '\n';
    o << weights[n] << '\t' << cweight[n] << '\t' ;
    o << weights[n].momentum() << '\n';
    o << "</to>\n";
  }
  o << "</backprop>\n";
  return o;
}
/*=======================================================================*/
/* void connection::backprop::adjust(bool a_set_err, bool a_epoch)
{
  double l_err, l_delta, l_sigma = 0;
  if (fromcon->is_output()) 
    l_sigma = trans->xfer(fromcon->fixed()) - xfer;
  else for (int n=0; n<tocount; n++)
  {
    l_err = tolist[n]->error();
    if (fabs(l_err) < 1.0e-120) l_err = 0;
    
    l_sigma += l_err * weights[n];
    if (a_epoch)
    {
      l_delta  = l_err * xfer * eta;
      
      weights[n] += weights[n].momentum() + l_delta;
      weights[n].momentum() = l_delta * lambda;
    }
  }
  l_sigma = trans->diff(xfer) * l_sigma;
  if (a_set_err) 
  {
    bias += eta * l_sigma;
    fromcon->set_error(l_sigma);
  }
  else
    fromcon->set_error(fromcon->error()+l_sigma);    
}

*/
/*=======================================================================*/
void connection::backprop::adjust(bool a_set_err, bool a_epoch)
{
  double l_err, l_delta, l_sigma = 0;
  if (fromcon->is_output()) 
    l_sigma = trans->xfer(fromcon->fixed()) - xfer;
  else for (int n=0; n<tocount; n++)
  {
    l_err = tolist[n]->error();
    if (fabs(l_err) < 1.0e-120) l_err = 0;
    
    l_sigma += l_err * weights[n];
    if (a_epoch)
    {
      l_delta  = l_err * xfer * eta;
      
      weights[n] += weights[n].momentum() + l_delta;
      weights[n].momentum() = l_delta * lambda;
    }
  }
  l_sigma = trans->diff(xfer) * l_sigma;
  if (a_set_err) 
  {
    bias += eta * l_sigma * trans->diff(bias+0.5);
    fromcon->set_error(l_sigma);
  }
  else
    fromcon->set_error(fromcon->error()+l_sigma);    
}


/*=======================================================================*/
istream &connection::common::get_info(GRID g, istream &i)
{
  char l_buf[256];
  i.getline(l_buf, sizeof(l_buf));   // </common>
  return i;
}
/*=======================================================================*/
ostream &connection::common::put_info(ostream &o)
{
  o << "<common>\n";
  o << "</common>\n";
  return o;
}
/*=======================================================================*/
void connection::common::adjust(bool a_set_err, bool a_epoch)
{
  double l_sigma = 0;
  
  if (fromcon->incount())
  {
    l_sigma = fromcon->fixed() - xfer;
    fromcon->set(xfer);
    fromcon->set_error(l_sigma/(double)fromcon->incount());
  }  
}

/*=======================================================================*/
/*=======================================================================*/
transfer g_deftrans(def_xfer, def_diff);
/*=======================================================================*/
transfer::transfer(transfer_func a_func,
		   transfer_func a_diff)
{
  xfer_func = a_func;
  diff_func = a_diff;
}
/*=======================================================================*/
transfer::~transfer()
{
}

/*=======================================================================*/
double transfer::diff(double a_excitation)
{
  return diff_func(a_excitation);
}

/*=======================================================================*/
double transfer::xfer(double a_total)
{
  return xfer_func(a_total);
}

/*=======================================================================*/
double tanh_xfer(double a_val)
{
  return (2 / (1 + exp(-2 * a_val))) - 1;
}

/*=======================================================================*/
//  fn(x) =  2 / (1 + exp(-2 * a_val))) - 1;
//  1 + fn(x) = 2 / (1 + exp(-2 * a_val)));
//  (1 + fn(x)) * (1 + exp(-2 * a_val))) = 2;
//  1 + exp(-2 * a_val)) = 2 / (1 + fn(x));
//  exp(-2 * a_val)) = (2 / (1 + fn(x)) - 1);
//  -2 * a_val = log(2 / (1 + fn(x)) - 1);
//  a_val = log(2 / (1 + fn(x)) - 1) + 2;
/*=======================================================================*/
double tanh_getval(double a_val)
{
  return log(2 / (1 + a_val) - 1) + 2;
}

/*=======================================================================*/
double tanh_diff(double a_val)
{
  return 1 - (a_val * a_val);
}

/*=======================================================================*/
void connection::connect(const paramlist &a_parms,
			 connection *a_target, 
			 transfer   *a_trans)
{
  b_unused = false;
  if (!m_node) 
  {
    if (a_trans==NULL) a_trans = &g_deftrans;
    make_newnode(a_parms, a_trans);
  }  
  m_node->connect(a_target);
}

/*=======================================================================*/
void connection::connect_layer_to(const paramlist &a_parms,
			       connection *a_target, 
			       transfer   *a_trans)
{
  connect(a_parms, a_target, a_trans);
  if (m_next) m_next->connect(a_parms, a_target, a_trans);
}

/*=======================================================================*/
void connection::connect_to_layer(const paramlist &a_parms,
			       connection *a_layer, 
			       transfer   *a_trans)
{
  b_unused = false;
  if (a_trans==NULL) a_trans = &g_deftrans;
  if (m_node == NULL) make_newnode(a_parms, a_trans);
  if (m_node != NULL) m_node->connect_to_layer(a_layer);  
  // add to existing connections
}

/*=======================================================================*/
void connection::connect_layers(const paramlist &a_parms,
				connection *a_layer, 
				transfer   *a_trans)
{
  connect_to_layer(a_parms, a_layer, a_trans);  // connect this to output layer
  if (m_next) m_next->connect_layers(a_parms, a_layer, a_trans);
}

/*=======================================================================*/
double connection::sigma(double a_total) 
{      // Return total error over layer
  if (!b_unused) a_total += (m_error * m_error);
  if (m_next) return m_next->sigma(a_total);
  return a_total * 0.5;
}
   
/*=======================================================================*/
void   connection::set(double a_value) 
{                     // Set Value for this
  b_unused = false; m_fixed = a_value; b_fixed = true; 
}
   
/*=======================================================================*/
void   connection::set(int n, double *a_vals) 
{             // Set Values for layer
  if (n>0 && a_vals) {
    set(a_vals[0]);
    if (m_next) m_next->set(--n, ++a_vals);
  }
  else rest_unused();
}
   
/*=======================================================================*/
void   connection::rest_unused() 
{                     // Set Values for layer
  set_unused(); if (m_next) m_next->rest_unused();
}

/*=======================================================================*/
void   connection::input() 
{        // copy input value 
  m_total = m_fixed; 
}

/*=======================================================================*/
void   connection::input(double a_value) 
{                  // set an input value
  set(a_value); m_total = a_value; b_input = true;
}

/*=======================================================================*/
void connection::input(int n, double *a_vals) 
{                  // input a set of values
  if (n > 0 && a_vals) 
  {
//    printf("in%i=%.2f\n", n, a_vals[0]);
    input(a_vals[0]);
    if (m_next) m_next->input(--n, ++a_vals);
  }
  else rest_unused();
}
   
/*=======================================================================*/
void connection::output(int n, double *a_vals) 
{            // set target values
  if (n > 0 && a_vals) 
  {
    if (!m_node->is_common()) set(a_vals[0]);
    if (m_next) m_next->output(--n, ++a_vals);
  }
  else rest_unused();
}

/*=======================================================================*/
connection *connection::find(char *a_tag) 
{ 
  if (!strcmp(tag(), a_tag)) return this;
  if (m_next) return m_next->find(a_tag);
  return NULL;
}
   
/*=======================================================================*/
// return n'th item in the layer
connection *connection::nth(int a_colno) 
{
  if (a_colno<=0) return this;
  return m_next->nth(a_colno-1);
}

/*=======================================================================*/
ostream& connection::put( ostream &o ) const
{
  o << tag() << '\n';
  o << m_rno << '\t' << m_cno << '\t' << m_fixed << '\t';
  o << b_unused << '\t' << b_fixed << '\t' << b_input << '\n';
  if (m_next) return m_next->put(o);
  return o;
}

/*=======================================================================*/
istream& connection::get( istream &i ) 
{
  char l_tag[64];
  i.getline(l_tag, sizeof(l_tag));
  tag(l_tag);
  i >> m_rno >> m_cno >> m_fixed;
  i >> b_unused >> b_fixed >> b_input;
//  printf("Tag:%s; unused=%i; fixed=%i; input=%i\n", 
//	 l_tag, b_unused, b_fixed, b_input);
  m_total = m_output = 0;
  i.getline(l_tag, sizeof(l_tag));
  if (m_next) return m_next->get(i);
  return i;
}
   
/*=======================================================================*/
   // for all connections in layer, dump neurode info
ostream& connection::put_info( ostream &o ) const 
{
  o << "<info>\n" << tag() << '\n';
  if (m_node) m_node->put_info(o);
  o << "</info>\n";
  if (m_next) return m_next->put_info(o);
  return o;
}

/*=======================================================================*/
   // for all connections in layer, read neurode info
istream& connection::get_info(GRID g, istream &i) 
{
  char l_buf[256];
  char l_neurtype[64];
  i.getline(l_buf, sizeof(l_buf));
  if (strcmp(l_buf, "<info>"))
  {
    if (!strcmp(l_buf, "</info>")) return i;
    fprintf(stderr, "Something _major_ is wrong. Expected <info>, got %s\n",
	    l_buf);
    return i;
  }
  i.getline(l_buf, sizeof(l_buf));
  if (strcmp(l_buf, tag()))
  {
    fprintf(stderr,"Tags do not match while reading connection info: %s<>%s\n",
	    l_buf, tag());
    return i;
  }
  
  i.getline(l_neurtype, sizeof(l_neurtype));
  if (!strcmp(l_neurtype, "</info>"))
  {
    if (m_next) m_next->get_info(g, i);
    return i;
  }
  
  if (!strcmp(l_neurtype, "<backprop>"))
    m_node = new backprop(this, &g_deftrans);
  else if (!strcmp(l_neurtype, "<common>"))
    m_node = new common(this, &g_deftrans);
  else if (!strcmp(l_neurtype, "<delta-bar-delta>"))
    m_node = new delta_bar_delta(this, &g_deftrans);
  else if (!strcmp(l_neurtype, "<backprop-decay>"))
    m_node = new backprop_decay(this, &g_deftrans);
  else if (!strcmp(l_neurtype, "<backprop_cw>"))
    m_node = new backprop_cw(this, &g_deftrans);
  else
  {
    fprintf(stderr, "Unknown neurode type %s\n", l_neurtype);
    return i;
  }
  m_node->get_info(g, i);
  i.getline(l_buf, sizeof(l_buf));     // </info>
  if (m_next) m_next->get_info(g, i);
  return i;
}
/*=======================================================================*/
connection::connection(int a_rno, char *label, int nconns)
{
  char l_tag[64];
  
  nconns--;
  m_cno = nconns;
  m_rno = a_rno;
  if (nconns) 
    m_next = new connection(a_rno, label, nconns);
  else
    m_next = NULL;
  
  m_tag       = NULL;
  m_node      = NULL;
  b_input     = false;
  b_fixed     = false;
  b_unused    = true;
  m_fixed     = 0;
  m_error     = 0;
  m_total     = 0;
  m_incount   = 0;
  m_monitor   = false;
  
  sprintf(l_tag, "%s(%i)", label, nconns);
  tag(l_tag);
}

/*=======================================================================*/
connection::~connection()
{
  if (m_node) delete m_node;
  if (m_next) delete m_next;
  m_next = NULL;
}

/*=======================================================================*/
// create a new neurode, depending on specified type
void connection::make_newnode(const paramlist &a_parms, transfer *a_trans)
{
  const char *l_type = a_parms.getparm("type", "bp");
  if (!strcmp(l_type, "bp"))
    m_node = new backprop(this, a_trans);
  else if (!strcmp(l_type, "dbd"))
    m_node = new delta_bar_delta(this, a_trans);
  else if (!strcmp(l_type, "common"))
    m_node = new common(this, a_trans);
  else if (!strcmp(l_type, "bpc"))
    m_node = new backprop_cw(this, a_trans);
  else if (!strcmp(l_type, "bp_decay"))
    m_node = new backprop_decay(this, a_trans);
  else
    m_node = new backprop(this, a_trans);
  m_node->set_params(a_parms);
}
   
/*=======================================================================*/
   // create a new neurode, depending on specified type
void connection::make_newnode(const paramlist &a_parms, 
			      connection      *a_layer, 
			      transfer        *a_trans)
{
  const char *l_type = a_parms.getparm("type", "bp");
  if (!strcmp(l_type, "bp"))
    m_node = new backprop(this, a_layer, a_trans);
  else if (!strcmp(l_type, "bpc"))
    m_node = new backprop_cw(this, a_layer, a_trans);
  else if (!strcmp(l_type, "dbd"))
    m_node = new delta_bar_delta(this, a_layer, a_trans);
  else if (!strcmp(l_type, "bp_decay"))
    m_node = new backprop_decay(this, a_layer, a_trans);
  else
    m_node = new backprop(this, a_layer, a_trans);
  m_node->set_params(a_parms);
}
/*=======================================================================*/
/*=======================================================================*/
double network::scale(double a_val, double a_min, double a_max)
{
  double d = a_max - a_min;  
  return (a_val - a_min) / d;
}

/*=======================================================================*/
double network::descale(double a_val, double a_min, double a_max)
{
  double d = a_max - a_min;  
  return a_val * d + a_min;
}

/*=======================================================================*/
void network::set_input(int a_layer, int n, double a_in[],
			double a_min, double a_max)
{
  double d = a_max - a_min, in[n];
  for (int i = 0; i < n; i++) in[i] = (a_in[i] - a_min) / d;
  r[a_layer]->input(n, in);
}

/*=======================================================================*/
void network::set_output(int a_layer, int n, double a_out[],
			 double a_min, double a_max)
{
  double d = a_max - a_min, out[n];
  for (int i = 0; i < n; i++) out[i] = (a_out[i] - a_min) / d;
  r[a_layer]->output(n, out);
}

/*=======================================================================*/
void network::set_output(int a_layer, transfer *a_trans) 
{
  if (a_trans==NULL) a_trans = &g_deftrans;
  r[a_layer]->connect_layers(parms, NULL, a_trans);
}

/*=======================================================================*/
void network::connect_layer_to(int a_from, connection *a_to, transfer *a_trans)
{
  if (a_trans==NULL) a_trans = &g_deftrans;
  r[a_from]->connect_layer_to(parms, a_to, a_trans);
}

/*=======================================================================*/
void network::connect_layers(int a_from, int a_to, transfer *a_trans) 
{
  if (a_trans==NULL) a_trans = &g_deftrans;
  r[a_from]->connect_layers(parms, r[a_to], a_trans);
}

/*=======================================================================*/
connection *network::rc(int a_rowno, int a_colno)
{
  connection *c;
  if ((c = row(a_rowno))==NULL) return c;
  return c->nth(a_colno);
}

/*=======================================================================*/
int network::row_id(char *a_rowtag)
{
  for (int i=0; i < nrows; i++) if (!strcmp(t[i], a_rowtag)) return i;
  return -1;
}

/*=======================================================================*/
connection *network::row(char *a_rowtag)
{
  for (int i=0; i < nrows; i++) if (!strcmp(t[i], a_rowtag)) return r[i];
  return NULL;
}

/*=======================================================================*/
int network::addrow(char *a_label, int a_width) 
{
  nrows++;
  r = (connection **)realloc(r, (nrows+1) * sizeof(r[0]));
  t = (char **)realloc(t, (nrows+1) * sizeof(t[0]));
  t[nrows-1] = strdup(a_label);
  r[nrows-1] = new connection(nrows-1, a_label, a_width);
  r[nrows] = NULL;
  return nrows-1;
}

/*=======================================================================*/
double network::learn()
{
  for (int n = 0; n < nrows; n++) r[n]->fire();
  for (int n = nrows; n > 0; n--) r[n-1]->adjust(true, true);
  double s = r[nrows-1]->sigma();
  return s;
}

/*=======================================================================*/
double network::learnbatch(bool a_last)
{
  for (int n = 0; n < nrows; n++) r[n]->fire();
  if (a_last)
  {
    for (int n = nrows; n > 0; n--) r[n-1]->adjust(false, true);
    for (int n = nrows; n > 0; n--) r[n-1]->adjust(true, false);
  }
  else
    for (int n = nrows; n > 0; n--) r[n-1]->adjust(false, false);
    
  double s = r[nrows-1]->sigma();
  return s;
}

/*=======================================================================*/
double network::learn(double *a_in, double *a_out)
{
  connection *in = r[0], *out = r[nrows-1];
  in->input(in->count(), a_in);
  out->set(out->count(), a_out);
  return learn();
}

/*=======================================================================*/
void network::docalc()                 // calc the outputs for the network
{
  for (int n = 0; n < nrows; n++) r[n]->fire();     
}

/*=======================================================================*/
void network::docalc(double a_in[])    // calc the outputs for the network
{
  r[0]->input(r[0]->count(), a_in);
  docalc();
}

/*=======================================================================*/
istream& network::get(istream &i) 
{
  char l_buf[20];
  i >> nrows;
  i.getline(l_buf, sizeof(l_buf));
  i >> parms;
  r = (connection **)realloc(r, (nrows+1) * sizeof(r[0]));
  t = (char **)realloc(t, (nrows+1) * sizeof(t[0]));
  i.getline(l_buf, sizeof(l_buf));
  if (!strcmp(l_buf, "<tags>"))
  {
    for (int p = 0; p < nrows; p++)
    {
      i.getline(l_buf, sizeof(l_buf));
      if (!strcmp(l_buf, "</tags>")) break;
      t[p] = strdup(l_buf);
    }
    if (strcmp(l_buf, "</tags>"))
      i.getline(l_buf, sizeof(l_buf));  // </tags>
  }
  else 
    for (int p = 0; p < nrows; p++) t[p] = strdup("undef");
  
  for (int p = 0; p < nrows; p++) i >> r[p];
  r[nrows] = NULL;
  i >> r;
  fprintf(stderr, "Number of rows=%i\n", nrows);
  return i;
}

/*=======================================================================*/
ostream& network::put(ostream &o) const 
{
  o << nrows << '\n' << parms;
  o << "<tags>\n";
  for (int p = 0; p < nrows; p++) o << t[p] << '\n';
  o << "</tags>\n";
  for (int p = 0; p < nrows; p++) o << r[p];
  o << r;
  return o;
}

/*========================================================================*/

istream& operator >> (istream& i, paramlist &p){p.get(i); return i; }
ostream& operator << (ostream& o, const paramlist &p){p.put(o); return o; }
/*========================================================================*/
// Read a network from a stream
istream& operator >> (istream& i, network &n) 
{
  char l_buf[256];
  i.getline(l_buf, sizeof(l_buf));
  if (!strcmp(l_buf, "<network>"))
  {
    n.get(i);
    i.getline(l_buf, sizeof(l_buf));
  }
  return i;
}

/*=======================================================================*/
ostream& operator << (ostream& o, const network &n) 
{
  o << "<network>\n"; n.put(o); o << "</network>\n";
  return o;
}

/*========================================================================*/
// Read a connection layer
istream& operator >> (istream& i, connection *&c) 
{ 
  char l_buf[256];
  int  n;
  i.getline(l_buf, sizeof(l_buf));
  if (!strcmp(l_buf, "<row>"))
  {
    i >> n;
    i.getline(l_buf, sizeof(l_buf));
    c = new connection(0, "", n);
    c->get(i);                 // read row information
    i.getline(l_buf, sizeof(l_buf));
  }
  else
    fprintf(stderr, "Expecting <row>, but got '%s'\n", l_buf);
  return i;
}

/*=======================================================================*/
ostream& operator << (ostream& o, const connection *c) 
{ 
  o << "<row>\n" << c->count() << '\n'; c->put(o); o << "</row>\n";
  return o;
}

/*========================================================================*/
// Read network settings and create connections
istream& operator >> (istream& i, GRID &g) 
{       // read architecture
  char l_buf[256];
  g_nconns = 0;
  
  i.getline(l_buf, sizeof(l_buf));
  if (!strcmp(l_buf, "<architecture>"))
  {  // For all rows and connection elements, recreate and connect neurodes
    i.precision(12);
    for (int n = 0; g[n]!=NULL; n++) g[n]->get_info(g, i); 
  }
  fprintf(stderr, "Number of connections made=%i\n", g_nconns);
  return i;
}

/*=======================================================================*/
ostream& operator << (ostream& o, const GRID &c)
{      // dump architecture
  o << "<architecture>\n";
  o.precision(12);
  for (int n = 0; c[n]!=NULL; n++) c[n]->put_info(o); 
  o << "</architecture>\n";
  return o;
}

/*=======================================================================*/
int xxxmain()
{
  network  n(paramlist("type=dbd"));
//  network  n(paramlist("type=bp","eta=0.75", "lambda=0.1"));
//  network  n(paramlist("type=bp_decay","eta=0.75","lambda=0.1"));
  double i[4][2] = {{0.0,0.0},{0.0,1.0},{1.0,0.0},{1.0,1.0}};
  double o[4][1] = {{0.0},{1.0},{1.0},{0.0}};
  
  int in, m1, out;

  in  = n.addrow("i", 2);
  m1  = n.addrow("m1", 3);
  out = n.addrow("o", 1);
  
  n.connect_layers(in,m1);
  n.connect_layers(in,out);
  n.connect_layers(m1,out);
  n.set_output(out);
  
  int p;
  for (p = 0; p < 500000; p++)
    if (n.learn(i[p%4], o[p%4]) < 1.0e-17) break;

  fprintf(stderr, "Iterations=%i\n", p);
  
  for (p = 0; p < 4; p++)
  {
    n.docalc(i[p]);
    fprintf(stderr, "%3.0f, %3.0f = %8.6f\n", i[p][0], i[p][1], n[0]);
  }
  return 0;
}

