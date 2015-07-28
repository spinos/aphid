class AStripedModel;
class BaseBuffer;
class StripeMap {
public:
    StripeMap();
    virtual ~StripeMap();
    
    void create(AStripedModel * mdl);
    void setLastIndex(unsigned x);
    void computeTetrahedronInStripe(float * dst, unsigned n);
protected:
    unsigned stripeBegin(unsigned i) const;
    unsigned * stripeBegins() const;
private:
    BaseBuffer * m_stripeIndices;
    unsigned m_numStripes;
    unsigned m_lastIndex;
};
