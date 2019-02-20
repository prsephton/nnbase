using namespace std;

#ifndef __nnbase_h__
#define __nnbase_h__
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef double (*transfer_func)(double a_val);
/*=========================================================================*\
 *  The transfer class allows one to very simply group a transfer function
 * and associated differential function.  The resulting object can be
 * plugged in when creating connections.
\*=========================================================================*/
class connection;
typedef connection ** GRID;
class transfer
{
 private :
  transfer_func     xfer_func;
  transfer_func     diff_func;
 public :
  transfer(transfer_func a_func, transfer_func a_diff);
  ~transfer();
  double xfer(double a_total);        // transfer for value
  double diff(double a_excitation);   // differential
};

/*=========================================================================*\
 * Parameter lists are used to maintain a very simple set of parameters
 * for use in assigning default values to various network components.
\*=========================================================================*/
class paramlist
{
  private :
   #define MAX_PARAM 10
   
   char *p[MAX_PARAM];
   char *pv[MAX_PARAM];
   int   np;
   void readparm(int n, const char *a_parm);
   
  public :
   paramlist::paramlist(const char *a_params);
   paramlist(char *params[]);
   paramlist(char *p1="type:bp", char *p2=NULL, char *p3=NULL, char *p4=NULL,
	     char *p5=NULL, char *p6=NULL, char *p7=NULL, char *p8=NULL, 
	     char *p9=NULL, char *p10=NULL, char *pmax=NULL);
   ~paramlist();
   paramlist &operator=(const paramlist &c);
   const char *operator[](const int i) const;
   const char *operator[](const char *s) const;
   const char *getparm(const char *a_parm, const char *a_default) const;
   istream& get(istream &i);
   ostream& put(ostream &o) const;
};

// Tell input streams about parameter lists
istream& operator >> (istream& i, paramlist &p);
// and output streams too.
ostream& operator << (ostream& o, const paramlist &p);

/*=========================================================================*\
 * The connection class represents a single input or output.  Layers are 
 * represented as linked lists of connections.  Each connection object has
 * an associated neurode which targets a list of connections.  Instead of a 
 * neurode maintaining a total, the connection does so.  Connections may
 * also maintain a fixed input or output value for the purposes of signal
 * propagation, or back propagation of error.
 * A connection has no knowledge as to the number of neurodes targeting it. 
 * Processing is done by firing each connection in turn, which effectively
 * fires the associated neurode.  The neurode copies the fixed input value
 * if available rather than the excitation total, and propagates the signal
 * to its list of targets by applying the appropriate weight.  Note that
 * the output layer connections are no different, as they also have associated
 * neurodes.  In the output layer, neurodes are responsible for comparing the
 * signal to the expected (fixed) signal and setting the error for the parent
 * connection.  Neurodes targeting this connection may then reference the
 * error value for the purpose of adjusting the weights.
\*=========================================================================*/
class connection
{
  private :
   int             m_rno;
   int             m_cno;
   double          m_error;
   double          m_total;
   double          m_fixed;
   double          m_output;
   bool            b_unused;
   bool            b_fixed;
   bool            b_input;
   connection     *m_next;
   int             m_incount;
   
   // Return number of connections remaining in this list
   int count(int a_first) const {      
     if (m_next) return m_next->count(a_first + 1);
     return a_first;
   }
   
  public :
   /*=====================================================================*/
   // Weights between a neurode and it's targets is represented as an
   // array of this class, rather than a simple array of doubles.
   /*=====================================================================*/
   class weight
   {
     private :
      double m_value;           // the weight value
      double m_eta;             // a weight specific learning rate
      double m_avg_slope;       // the average error slope
      double m_momentum;        // momentum for this weight
     public  :
      weight(int    a_val) { m_avg_slope = 0; m_eta=0.25; m_value = a_val; }
      weight(double a_val) { m_avg_slope = 0; m_eta=0.25; m_value = a_val; }
      weight() { m_avg_slope = 0; m_eta=0.25; 
	m_value = (1.0-((rand() * 2.0/(double)RAND_MAX)));
      }
      double operator *(const double a_val)  { return m_value * a_val; }
      double operator +(const double a_val)  { return m_value + a_val; }
      double operator =(const int a_val)     { return m_value = a_val; }
      double operator =(const double a_val)  { return m_value = a_val; }
      double operator ==(const double a_val) { return m_value == a_val; }
      double operator >(const double a_val)  { return m_value > a_val; }
      double operator <(const double a_val)  { return m_value < a_val; }
      double operator >=(const double a_val) { return m_value >= a_val; }
      double operator <=(const double a_val) { return m_value <= a_val; }
      operator double&() { return m_value; }
      double &eta()      { return m_eta;   }
      double &slope()    { return m_avg_slope; }
      double &momentum() { return m_momentum; }
   };
   
   /*=====================================================================*\
    * A virtual base class defines the basic neurode.
   \*=====================================================================*/
   class neurode
   {
     public  :
      connection **tolist;          // targeted connections
      connection  *fromcon;         // connection to this neurode
      transfer    *trans;           // transfer and differential functions
      int          tocount;         // number of targeted connections
      double       xfer;            // squashed total excitation
      double       bias;            // node bias
      double       eta;             // node learning rate
      double       lambda;          // momentum adjustment factor
      weight      *weights;         // weights to target connections
      weight      *cweight;         // counterweights
      
      neurode(connection *a_from, transfer *a_trans);
      neurode(connection *a_from, connection *a_to, transfer *a_trans);
      virtual ~neurode();
      void connect(connection *a_target);
      void connect_to_layer(connection *a_out);
      double      psignal() { return xfer; }
      double      isignal() { return trans->xfer(fromcon->output()-bias); }
      double      pbias()   { return trans->xfer(bias); }
      int         tcount()  { return tocount; }
      connection *tconn(int n) {
	if (n < tocount) return tolist[n]; return NULL; }
      double tweight(int n) {
	if (n < tocount) return trans->xfer(weights[n]+cweight[n]); return 0.0; }
      virtual void set_params(const paramlist &a_parm) {};
      virtual void fire();
      virtual bool is_common() { return false; }
      
      virtual void adjust(bool a_set_err, bool a_epoch) = 0;
      virtual ostream &put_info(ostream &o) = 0;
      virtual istream &get_info(GRID g, istream &i) = 0;
   };
   
   /*=====================================================================*/
   // Specialisations of the neurode base class start here.
   /*=====================================================================*/
   // Normal backpropagation rule with momentum constant
   class backprop : public neurode
   {
     public :
      backprop(connection *a_from, transfer *a_trans) 
	: neurode(a_from, a_trans) {};
      backprop(connection *a_from, connection *a_to, transfer *a_trans) 
	: neurode(a_from, a_to, a_trans) {};
      virtual void set_params(const paramlist &a_parm) {
	eta     = atof(a_parm.getparm("eta",    "0.75"));
        lambda  = atof(a_parm.getparm("lambda", "0.01"));
      };
      virtual void adjust(bool a_set_err, bool a_epoch);
      virtual istream &get_info(GRID g, istream &i);
      virtual ostream &put_info(ostream &o);
   };
   
   // Calculates difference from mean inputs as error.
   // No fixed output value exists.  This neurode type
   // tries to synchronise between inputs.
   class common : public neurode
   {
     public :
      common(connection *a_from, transfer *a_trans) 
	: neurode(a_from, a_trans) {};
      
      virtual void adjust(bool a_set_err, bool a_epoch);
      
      virtual bool is_common() { return true; }
      virtual istream &get_info(GRID g, istream &i);
      virtual ostream &put_info(ostream &o);
   };
   
   // Backpropagation with counterweight adjustments
   class backprop_cw : public neurode
   {
     private :
      double       theta, kappa, max_eta, decay;
     public :
      virtual void fire();
      backprop_cw(connection *a_from, transfer *a_trans) 
	: neurode(a_from, a_trans) { 
	  theta = 0.7; kappa=0.1; max_eta=10; decay=1.0e-3; 
	}
      
      backprop_cw(connection *a_from, connection *a_to, transfer *a_trans) 
	: neurode(a_from, a_to, a_trans) { 
	  theta = 0.7; kappa=0.1; max_eta=10; decay=1.0e-3; 
	}
      virtual void set_params(const paramlist &a_parm) {
	eta     = atof(a_parm.getparm("eta",    "0.75"));
	theta   = atof(a_parm.getparm("theta",  "0.45"));
	kappa   = atof(a_parm.getparm("kappa",  "0.1"));
	decay   = atof(a_parm.getparm("decay",  "1.0e-9"));
        lambda  = atof(a_parm.getparm("lambda", "0.01"));
        max_eta = atof(a_parm.getparm("etamax", "10.0"));
/*	
	printf("eta=%.2g; ", eta);
	printf("theta=%.2g; ", theta);
	printf("kappa=%.2g; ", kappa);
	printf("decay=%.2g; ", decay);
	printf("lambda=%.2g; ", lambda);
	printf("max_eta=%.2g\n", max_eta);
 */
      };
      virtual void adjust(bool a_set_err, bool a_epoch);
      virtual istream &get_info(GRID g, istream &i);
      virtual ostream &put_info(ostream &o);
   };
   
   /*=====================================================================*/
   // Delta-Bar-Delta learning rule. Like backprop, but with variable 
   // learning rate based on oscillation.
   class delta_bar_delta : public neurode
   {
     private :
      double       theta, kappa, max_eta, decay;
     public :
      delta_bar_delta(connection *a_from, transfer *a_trans) 
	: neurode(a_from, a_trans) { 
	  theta = 0.7; kappa=0.1; max_eta=10; decay=0.1; 
	}
      delta_bar_delta(connection *a_from, connection *a_to, transfer *a_trans) 
	: neurode(a_from, a_to, a_trans) { 
	  theta = 0.7; kappa=0.1; max_eta=10; decay=0.1; 
	}
      virtual void set_params(const paramlist &a_parm) {
	eta     = atof(a_parm.getparm("eta",    "0.75"));
        lambda  = atof(a_parm.getparm("lambda", "0.01"));
	theta   = atof(a_parm.getparm("theta",  "0.7"));
	kappa   = atof(a_parm.getparm("kappa",  "0.1"));
	decay   = atof(a_parm.getparm("decay",  "0.1"));
        max_eta = atof(a_parm.getparm("etamax", "10.0"));
      };
      virtual void adjust(bool a_set_err, bool a_epoch);
      virtual istream &get_info(GRID g, istream &i);
      virtual ostream &put_info(ostream &o);
   };
   
   /*=====================================================================*/
   // Backpropagation with decay and momentum constant
   class backprop_decay : public neurode
   {
     private :
      double decay;   // weight decay constant
     public :
      backprop_decay(connection *a_from, transfer *a_trans) 
	: neurode(a_from, a_trans) { decay = 0.01; }
      backprop_decay(connection *a_from, connection *a_to, transfer *a_trans) 
	: neurode(a_from, a_to, a_trans) { decay = 0.01; }
      virtual void set_params(const paramlist &a_parm) {
	eta     = atof(a_parm.getparm("eta",    "0.75"));
	decay   = atof(a_parm.getparm("decay",  "1.0e-12"));
        lambda  = atof(a_parm.getparm("lambda", "0.01"));
      };
      virtual void adjust(bool a_set_err, bool a_epoch);
      virtual istream &get_info(GRID g, istream &i);
      virtual ostream &put_info(ostream &o);
   };
   
   
  private :
   neurode        *m_node;
   char           *m_tag;
   bool            m_monitor;
   
   void   firecon()    { if (m_node) m_node->fire(); }
   void   adjust_con(bool a_set_err, bool a_epoch) { 
     if (m_node) m_node->adjust(a_set_err, a_epoch); }
   
   /*=====================================================================*/
   // Public connection class definition. A connection may point to a next
   // connection forming a row, and will have an associated neurode.  The
   // neurode has a list of target connections, and a reference to it's
   // associated parent connection.
  public  :
   connection(int a_rno, char *label, int nconns);
   ~connection();
   
   double sigma(double a_total);      // Return total error over layer
   
   ostream& put( ostream &o ) const;
   istream& get( istream &i );
   
   // for all connections in layer, dump neurode info
   ostream& put_info( ostream &o ) const;
   istream& get_info( GRID g, istream &i );
   
   const char *tag() const { return m_tag; }       // Connection tag
   void   tag(char *a_tag) {if (m_tag) free(m_tag); m_tag = strdup(a_tag); }
   int    row() { return m_rno; }
   int    col() { return m_cno; }
   void   set(double a_value);                     // Set Value for this
   void   set(int n, double *a_vals);              // Set Values for layer
   void   set_unused() {  b_unused = true; }       // Set Unused node
   void   rest_unused();                           // rest of layer is unused
   bool   is_input() { return b_input; }           // is this an input?
   bool   is_output() { return b_fixed && !b_input; } // is this an output?
   void   input();                                 // copy input value
   void   input(double a_value);                   // set an input value
   void   input(int n, double *a_vals);            // input a set of values
   void   output(int n, double *a_vals);           // set target values
   double fixed() const { return m_fixed; }        // Get Fixed Value
   void   monitor() const {  
     if (m_monitor) {  printf("%s:total=%f\n", tag(), m_total); } }
   
   void   mon_output() const {
     if (m_monitor) {  printf("%s:output=%f\n", tag(), m_output); } }
   
   void   monitor(bool a_switch) { 
     m_monitor = a_switch; 
     if (m_next) m_next->monitor(a_switch);
   }
   double total() const { monitor(); return m_total; }  // Get Output Value
   int    incount() const { return m_incount; }   // Number of updates
   int    count() const { return count(1); }   // Return number of connections
   // Fire the layer
   void   fire() { firecon(); if (m_next) m_next->fire(); }
   // Propagate error backward & adjust
   void   adjust(bool a_set_err, bool a_epoch) { 
     adjust_con(a_set_err, a_epoch); 
     if (m_next) m_next->adjust(a_set_err, a_epoch); }
   
   double operator[](const int i) { 
     if (i>0 && m_next) return (*m_next)[i-1];
     if (i) return 0.0;
     return output();
   }
   
   double sigma() { return sigma(0); }    // Return error on layer
   
   // For output nodes, the error is calced as the difference between 
   // target and the transfer total, but only while learning...
   double error()  { if (b_unused) return 0; return m_error; }
   // Is the connection marked as inactive?
   bool ignore() { return b_unused; }
   // Retrieve the output for a node.
   double output() { return m_output; }
   // Add to total for this connection
   void   update(double a_update) { m_incount++; m_total += a_update; }
   // Same story for error- this makes a total error for all but the output
   void   set_error(double a_err) { 
//     if (is_output()) printf("%s: err=%.4g\n", tag(), a_err);
     m_incount = 0;
     m_error = a_err; 
   }
   // Set the output=total, and total to 0
   void   do_output(double a_bias) { 
     m_output = m_total + a_bias; mon_output(); m_total=0; }
   // Return the next connection in the layer
   connection *next() { return m_next; }
   // find the element corresponding to the tag, or NULL
   connection *find(char *a_tag);
   // return n'th item in the layer
   connection *nth(int a_colno);
   // return the total input signal to this
   double isignal() { if (m_node) return m_node->isignal(); return 0.0; }
   // return the total propagation signal from this
   double psignal() { if (m_node) return m_node->psignal(); return 0.0; }
   // return the number of targets for this connection
   double pbias() { if (m_node) return m_node->pbias(); return 0.0; }
   // return the number of targets for this connection
   int tocount() { if (m_node) return m_node->tcount(); return 0; }
   // return the n'th connection targeted by this
   connection *toconn(int n) { if (m_node) return m_node->tconn(n); return NULL; }
   // return the weighting for the n'th connection targeted by this
   double toweight(int n) {if (m_node) return m_node->tweight(n); return 0.0; }
   // create a new neurode, depending on specified type
   void make_newnode(const paramlist &a_parms, transfer *a_trans);
   // create a new neurode, depending on specified type
   void make_newnode(const paramlist &a_parms, 
		     connection      *a_layer, 
		     transfer        *a_trans);
   // Connect this to a single specified target
   void connect(const paramlist &a_parms,
		connection      *a_target,
		transfer        *a_trans = NULL);
   
   // Connect this layer to a single target element
   void connect_layer_to(const paramlist &a_parms,
		      connection *a_target, 
		      transfer   *a_trans = NULL);
   // connect this element to the specified output layer
   void connect_to_layer(const paramlist &a_parms,
			 connection      *a_layer, 
			 transfer        *a_trans = NULL);
   // Connect this layer to the specified layer with transfer
   void connect_layers(const paramlist &a_parms,
		       connection *a_layer, 
		       transfer   *a_trans = NULL);
};

/*=======================================================================*/
// A grid is an array or rows.  The number of items in each row is not
// necessarily constant.  The array or rows is terminated by a NULL.
/*=======================================================================*/
typedef connection ** GRID;
// How do we output a grid to a stream?
istream& operator >> (istream& i, GRID &c);
// How do we output a row to a stream?
istream& operator >> (istream& i, connection *&c);
// How do we input a grid from a stream?
ostream& operator << (ostream& o, const GRID &c);
// How do we input a row from a stream?
ostream& operator << (ostream& o, const connection *c);

/*=======================================================================*/
/*=======================================================================*/
class network
{
   GRID         r;
   char       **t;
   int          nrows;
   paramlist    parms;
  public :
   network() {r = NULL; nrows = 0; t = NULL; }
   network(const paramlist &a_parms) { 
     parms = a_parms; r = NULL; nrows = 0; t = NULL; }
   ~network() { for (int n = 0; n < nrows; n++) delete r[n]; }
   // Retrieve a parameter by name
   const char *param(const char *a_name){ return parms[a_name]; }
   void params(const paramlist &a_parms){ 
//     fprintf("1: params=%s\n", a_parms.c_str());
     parms = a_parms; }
   void params(const char *a_parms){ 
//     fprintf(stderr, "2: params=%s\n", a_parms);
     parms = paramlist(a_parms); }
   
   // Add a row of connections
   int addrow(char *a_label, int a_width);
   // Fully connect two layers 
   void connect_layers(int a_from, int a_to, transfer *a_trans = NULL);
   // Connect from a layer to a single target
   void connect_layer_to(int a_from, connection *a_to, transfer *a_trans=NULL);
   // Return the tag for a row number
   const char *tag(int a_rowno) { return a_rowno < nrows?t[a_rowno]:""; }
   const char *tag(int a_rowno, int a_cno) { 
     if (a_rowno >= nrows) return "";
     if (a_cno >= r[a_rowno]->count()) return "";
     return r[a_rowno]->nth(a_cno)->tag();
   }
   // Return the row number for a tag, or -1
   int row_id(char *a_rowtag);
   // Return the number of rows
   int rows() { return nrows; }
   // Return the row indicated by the row number, or NULL
   connection *row(int a_rowno) { return a_rowno < nrows?r[a_rowno]:NULL; }
   // Return the row identified by the row tag, or NULL
   connection *row(char *a_rowtag);
   // Return the grid connection ar row[rowno] and column[colno]
   connection *rc(int a_rowno, int a_colno);
   // Set (n) input values for a layer.  The rest are marked unused
   void set_input( int a_layer, int n, double a_in[], 
		  double a_min, double a_max);
   // Set up to (n) output values in the layer, marking the rest unused
   void set_output(int a_layer, int n, double a_out[],
		   double a_min, double a_max);
   // Indicates final layer in network.
   void set_output(int a_layer, transfer *a_trans = NULL);
   // Set input/output and perform learning cycle assuming row0=<in>, n=<out>
   double learn(double a_in[], double a_out[]);  // use default inp/out
   // Perform learning cycle.  In/Out must be preset.
   double learn();                 // do not assume row0/row[n-1]
   double learnbatch(bool a_last);   // Learn, but adjust only with last val
   // Perform calculation from inputs to outputs
   void   docalc();                // calc the outputs for the network
   // Set layer0 to inputs and perform one calculation.
   void   docalc(double a_in[]);   // set default input layer and calc
   // scale a_val between min and max.
   double scale(double a_val, double a_min, double a_max);
   // reverse scale a_val between min and max.
   double descale(double a_val, double a_min, double a_max);
   // Return current network output for layer(n-1) for output <o>
   double operator[](const int o) { return (*(r[nrows-1]))[o]; }   
   // Read a network from an input stream
   istream& get(istream &i);
   // Write a network to an output stream
   ostream& put(ostream &o) const;
};

// Tell input streams about networks
istream& operator >> (istream& i, network &n);
// Tell output streams about networks
ostream& operator << (ostream& o, const network &n);

#endif
